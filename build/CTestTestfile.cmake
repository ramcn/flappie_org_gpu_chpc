# CMake generated Testfile for 
# Source directory: /home/chakenal/flappie_org_gpu
# Build directory: /home/chakenal/flappie_org_gpu/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unittest "/home/chakenal/flappie_org_gpu/build/flappie_unittest")
set_tests_properties(unittest PROPERTIES  WORKING_DIRECTORY "/home/chakenal/flappie_org_gpu/src/test/")
add_test(test_flappie_call "flappie" "/home/chakenal/flappie_org_gpu/reads")
add_test(test_flappie_licence "flappie" "--licence")
add_test(test_flappie_license "flappie" "--license")
add_test(test_flappie_help "flappie" "--help")
add_test(test_flappie_version "flappie" "--version")
add_test(test_runnie_call "runnie" "/home/chakenal/flappie_org_gpu/reads")
add_test(test_runnie_licence "runnie" "--licence")
add_test(test_runnie_license "runnie" "--license")
add_test(test_runnie_help "runnie" "--help")
add_test(test_runnie_version "runnie" "--version")
