/*  Copyright 2018 Oxford Nanopore Technologies, Ltd */

/*  This Source Code Form is subject to the terms of the Oxford Nanopore
 *  Technologies, Ltd. Public License, v. 1.0. If a copy of the License
 *  was not  distributed with this file, You can obtain one at
 *  http://nanoporetech.com
 */

#include <dirent.h>
#include <glob.h>
#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <strings.h>

#include "decode.h"
#include "fast5_interface.h"
#include "layers.h"
#include "networks.h"
#include "flappie_common.h"
#include "flappie_licence.h"
#include "flappie_output.h"
#include "flappie_stdlib.h"
#include "flappie_structures.h"
#include "util.h"
#include "version.h"

#if !defined(FLAPPIE_VERSION)
#    define FLAPPIE_VERSION "unknown"
#endif
const char *argp_program_version = "runnie " FLAPPIE_VERSION;
const char *argp_program_bug_address = "<tim.massingham@nanoporetech.com>";

// Doesn't play nice with other headers, include last
#include <argp.h>


extern const char *argp_program_version;
extern const char *argp_program_bug_address;
static char doc[] = "Runnie basecaller -- basecall from raw signal";
static char args_doc[] = "fast5 [fast5 ...]";
static struct argp_option options[] = {
    //{"format", 'f', "format", 0, "Format to output reads (FASTA or SAM)"},
    {"delta", 'd', "factor", 0, "Using delta samples model with scaling factor"},
    {"limit", 'l', "nreads", 0, "Maximum number of reads to call (0 is unlimited)"},
    //{"model", 'm', "name", 0, "Model to use (\"help\" to list)"},
    {"output", 'o', "filename", 0, "Write to file rather than stdout"},
    {"prefix", 'p', "string", 0, "Prefix to append to name of each read"},
    {"temperature", 7, "factor", 0, "Temperature for weights"},
    {"trim", 't', "start:end", 0, "Number of samples to trim, as start:end"},
    {"viterbi", 'v', 0, 0, "Use viterbi decoding only"},
    {"no-viterbi", 8, 0, OPTION_ALIAS, "Use forward-backward followed by viterbi"},
    {"fb", 9, 0, OPTION_ALIAS, "Use forward-backward followed by viterbi"},
    //{"trace", 'T', "filename", 0, "Dump trace to HDF5 file"},
    {"licence", 10, 0, 0, "Print licensing information"},
    {"license", 11, 0, OPTION_ALIAS, "Print licensing information"},
    {"segmentation", 3, "chunk:percentile", 0, "Chunk size and percentile for variance based segmentation"},
    //{"hdf5-compression", 12, "level", 0,
    // "Gzip compression level for HDF5 output (0:off, 1: quickest, 9: best)"},
    //{"hdf5-chunk", 13, "size", 0, "Chunk size for HDF5 output"},

    {"uuid", 14, 0, 0, "Output UUID"},
    {"no-uuid", 15, 0, OPTION_ALIAS, "Output read file"},
    {0}
};


#define DEFAULT_MODEL RUNNIE_NEWMODEL_R941_NATIVE

struct arguments {
    int compression_level;
    int compression_chunk_size;
    float delta;
    char * trace;
    enum flappie_outformat_type outformat;
    int limit;
    enum model_type model;
    FILE * output;
    char * prefix;
    float temperature;
    int trim_start;
    int trim_end;
    int varseg_chunk;
    float varseg_thresh;
    bool viterbi_only;
    char ** files;
    bool uuid;
};

static struct arguments args = {
    .compression_level = 1,
    .compression_chunk_size = 200,
    .delta = 1.0f,
    .trace = NULL,
    .limit = 0,
    .model = DEFAULT_MODEL,
    .output = NULL,
    .outformat = FLAPPIE_OUTFORMAT_FASTA,
    .prefix = "",
    .temperature = 1.0f,
    .trim_start = 200,
    .trim_end = 10,
    .varseg_chunk = 100,
    .varseg_thresh = 0.0f,
    .viterbi_only = false,
    .files = NULL,
    .uuid = true
};


void fprint_flappie_models(FILE * fh, enum model_type default_model){
    if(NULL == fh){
        return;
    }

    for(size_t mdl=0 ; mdl < flappie_nmodel ; mdl++){
        fprintf(fh, "%10s : %s  %s\n", flappie_model_string(mdl), flappie_model_description(mdl),
                                      (default_model == mdl) ? "(default)" : "");
    }
}


static error_t parse_arg(int key, char * arg, struct  argp_state * state){
    int ret = 0;
    char * next_tok = NULL;

    switch(key){
    /*case 'f':
        args.outformat = get_outformat(arg);
        if(FLAPPIE_OUTFORMAT_INVALID == args.outformat){
            errx(EXIT_FAILURE, "Unrecognised output format \"%s\".", arg);
        }
        break;*/
    case 'd':
        args.delta = atof(arg);
        assert(args.delta > 0.0f);
        break;
    case 'l':
        args.limit = atoi(arg);
        assert(args.limit > 0);
        break;
    /*
    case 'm':
        if(0 == strcasecmp(arg, "help")){
            fprint_flappie_models(stdout, DEFAULT_MODEL);
            exit(EXIT_SUCCESS);
        }
        args.model = get_flappie_model_type(arg);
        if(FLAPPIE_MODEL_INVALID == args.model){
            fprintf(stdout, "Invalid Flappie model \"%s\".\n", arg);
            fprint_flappie_models(stdout, DEFAULT_MODEL);
            exit(EXIT_FAILURE);
        }
        break;*/
    case 'o':
        args.output = fopen(arg, "w");
        if(NULL == args.output){
            errx(EXIT_FAILURE, "Failed to open \"%s\" for output.", arg);
        }
        break;
    case 'p':
        args.prefix = arg;
        break;
    case 't':
        args.trim_start = atoi(strtok(arg, ":"));
        next_tok = strtok(NULL, ":");
        if(NULL != next_tok){
            args.trim_end = atoi(next_tok);
        } else {
            args.trim_end = args.trim_start;
        }
        assert(args.trim_start >= 0);
        assert(args.trim_end >= 0);
        break;
    case 'v':
        args.viterbi_only = true;
        break;
    /*
    case 'T':
        args.trace = arg;
        break;*/
    case 3:
        args.varseg_chunk = atoi(strtok(arg, ":"));
        next_tok = strtok(NULL, ":");
        if(NULL == next_tok){
            errx(EXIT_FAILURE, "--segmentation should be of form chunk:percentile");
        }
        args.varseg_thresh = atof(next_tok) / 100.0;
        assert(args.varseg_chunk >= 0);
        assert(args.varseg_thresh > 0.0 && args.varseg_thresh < 1.0);
        break;
    case 7:
	args.temperature = atof(arg);
	assert(isfinite(args.temperature) && args.temperature > 0.0f);
        break;
    case 8:
        args.viterbi_only = false;
        break;
    case 9:
        args.viterbi_only = false;
        break;
    case 10:
    case 11:
        ret = fputs(flappie_licence_text, stdout);
        exit((EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE);
        break;
    /*
    case 12:
        args.compression_level = atoi(arg);
        assert(args.compression_level >= 0 && args.compression_level <= 9);
        break;
    case 13:
        args.compression_chunk_size = atoi(arg);
        assert(args.compression_chunk_size > 0);
        break;*/
    case 14:
        args.uuid = true;
        break;
    case 15:
        args.uuid = false;
        break;
    case ARGP_KEY_NO_ARGS:
        argp_usage (state);
        break;

    case ARGP_KEY_ARG:
        args.files = &state->argv[state->next - 1];
        state->next = state->argc;
        break;

    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}


static struct argp argp = {options, parse_arg, args_doc, doc};


static void calculate_post(char * filename, enum model_type model){
    RETURN_NULL_IF(NULL == filename, );

    raw_table rt = read_raw(filename, true);
    RETURN_NULL_IF(NULL == rt.raw, );

    rt = trim_and_segment_raw(rt, args.trim_start, args.trim_end, args.varseg_chunk, args.varseg_thresh);
    RETURN_NULL_IF(NULL == rt.raw, );

    if( args.delta == 0.0f){
        medmad_normalise_array(rt.raw + rt.start, rt.end - rt.start);
    } else {
        difference_array(rt.raw + rt.start, rt.end - rt.start);
        shift_scale_array(rt.raw + rt.start, rt.end - rt.start, 0.0, args.delta);
    }

    flappie_matrix trans_weights = calculate_transitions(rt, args.temperature, model);
    if (NULL == trans_weights) {
        free_raw_table(&rt);
        return;
    }

    const size_t nblock = trans_weights->nc;
    const size_t nparam = trans_weights->nr;
    const size_t nbase = nbase_from_crf_runlength_nparam(nparam);
    int * path = calloc(nblock + 2, sizeof(int));

    float score = NAN;


    flappie_matrix transpost = trans_weights;
    if(!args.viterbi_only){
        transpost = transpost_crf_runlength(trans_weights);
        free(trans_weights);
    }
    score = decode_crf_runlength(transpost, path);
    fprintf(args.output, "# %s\n", rt.uuid);

    {
        int dwell = 1;
        int last_blk = -1;
        for(size_t blk=0 ; blk < nblock ; blk++){
            if(path[blk] >= nbase){
                // No new base emitted, short circuit
                dwell += 1;
                continue;
            }

            //  New base
            if(last_blk >= 0){
                // If a base has already been called, emit run
                const size_t offset = last_blk * transpost->stride;
                const int base = path[last_blk];
                const float shape = transpost->data.f[offset + base];
                const float scale = transpost->data.f[offset + nbase + base];
                fprintf(args.output, "%c\t%f\t%f\t%d\n",
                        basechar(base), shape, scale, dwell);
            }
            last_blk = blk;
            dwell = 1;
        }
        if(last_blk >= 0){
            // Emit final base and run, if any
            const size_t offset = last_blk * transpost->stride;
            const int base = path[last_blk];
            const float shape = transpost->data.f[offset + base];
            const float scale = transpost->data.f[offset + nbase + base];
            fprintf(args.output, "%c\t%f\t%f\t%d\n",
                    basechar(base), shape, scale, dwell);
        }
    }

    transpost = free_flappie_matrix(transpost);
    free(path);
    free_raw_table(&rt);
}


int main(int argc, char * argv[]){
    argp_parse(&argp, argc, argv, 0, 0, NULL);
    if(NULL == args.output){
        args.output = stdout;
    }

    hid_t hdf5out = open_or_create_hdf5(args.trace);


    int nfile = 0;
    for( ; args.files[nfile] ; nfile++);

    int reads_started = 0;
    const int reads_limit = args.limit;

    for(int fn=0 ; fn < nfile ; fn++){
        if(reads_limit > 0 && reads_started >= reads_limit){
            continue;
        }
        //  Iterate through all files and directories on command line.
        glob_t globbuf;
        {
            // Find all files matching commandline argument using system glob
            const size_t rootlen = strlen(args.files[fn]);
            char * globpath = calloc(rootlen + 9, sizeof(char));
            memcpy(globpath, args.files[fn], rootlen * sizeof(char));
            {
                DIR * dirp = opendir(args.files[fn]);
                if(NULL != dirp){
                    // If filename is a directory, add wildcard to find all fast5 files within it
                    memcpy(globpath + rootlen, "/*.fast5", 8 * sizeof(char));
                    closedir(dirp);
                }
            }
            int globret = glob(globpath, GLOB_NOSORT, NULL, &globbuf);
            free(globpath);
            if(0 != globret){
                if(GLOB_NOMATCH == globret){
                    warnx("File or directory \"%s\" does not exist or no fast5 files found.", args.files[fn]);
                }
                globfree(&globbuf);
                continue;
            }
        }

        for(size_t fn2=0 ; fn2 < globbuf.gl_pathc ; fn2++){
            if(reads_limit > 0 && reads_started >= reads_limit){
                continue;
            }
            reads_started += 1;

            char * filename = globbuf.gl_pathv[fn2];
            calculate_post(filename, args.model);
        }
        globfree(&globbuf);
    }


    if (hdf5out >= 0) {
        H5Fclose(hdf5out);
    }

    if(stdout != args.output){
        fclose(args.output);
    }

    return EXIT_SUCCESS;
}
