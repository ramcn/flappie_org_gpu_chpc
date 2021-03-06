# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/chakenal/xilinx/SDK/2018.2/tps/lnx64/cmake-3.3.2/bin/cmake

# The command to remove a file.
RM = /home/chakenal/xilinx/SDK/2018.2/tps/lnx64/cmake-3.3.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chakenal/flappie_org_gpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chakenal/flappie_org_gpu/build

# Include any dependencies generated for this target.
include CMakeFiles/flappie_static.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/flappie_static.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/flappie_static.dir/flags.make

# Object files for target flappie_static
flappie_static_OBJECTS =

# External object files for target flappie_static
flappie_static_EXTERNAL_OBJECTS = \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/decode.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/layers.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/networks.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/nnfeatures.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/flappie_common.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/flappie_matrix.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/flappie_output.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/flappie_structures.c.o" \
"/home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_objects.dir/src/util.c.o"

libflappie.a: CMakeFiles/flappie_objects.dir/src/decode.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/layers.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/networks.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/nnfeatures.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/flappie_common.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/flappie_matrix.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/flappie_output.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/flappie_structures.c.o
libflappie.a: CMakeFiles/flappie_objects.dir/src/util.c.o
libflappie.a: CMakeFiles/flappie_static.dir/build.make
libflappie.a: CMakeFiles/flappie_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chakenal/flappie_org_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking C static library libflappie.a"
	$(CMAKE_COMMAND) -P CMakeFiles/flappie_static.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flappie_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/flappie_static.dir/build: libflappie.a

.PHONY : CMakeFiles/flappie_static.dir/build

CMakeFiles/flappie_static.dir/requires:

.PHONY : CMakeFiles/flappie_static.dir/requires

CMakeFiles/flappie_static.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/flappie_static.dir/cmake_clean.cmake
.PHONY : CMakeFiles/flappie_static.dir/clean

CMakeFiles/flappie_static.dir/depend:
	cd /home/chakenal/flappie_org_gpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chakenal/flappie_org_gpu /home/chakenal/flappie_org_gpu /home/chakenal/flappie_org_gpu/build /home/chakenal/flappie_org_gpu/build /home/chakenal/flappie_org_gpu/build/CMakeFiles/flappie_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/flappie_static.dir/depend

