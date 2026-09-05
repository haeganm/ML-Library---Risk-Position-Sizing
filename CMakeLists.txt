cmake_minimum_required(VERSION 3.21)
project(mlrisk VERSION 3.4.1 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(MLRISK_WERROR "Treat warnings as errors" ${PROJECT_IS_TOP_LEVEL})
option(MLRISK_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(MLRISK_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(MLRISK_BUILD_BENCH "Build the benchmark" OFF)
option(MLRISK_INSTALL "Generate install and export rules" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)

if(MSVC)
    # /fp:contract- forbids fused multiply-add the way -ffp-contract=off does
    set(MLRISK_COMPILE_OPTIONS /W4 /fp:precise /fp:contract- $<$<BOOL:${MLRISK_WERROR}>:/WX>)
else()
    # -ffp-contract=off: no fused multiply-add, so results match across
    # compilers and architectures to the last bit
    set(MLRISK_COMPILE_OPTIONS -Wall -Wextra -Wpedantic -ffp-contract=off
        $<$<BOOL:${MLRISK_WERROR}>:-Werror>)
endif()

configure_file(include/mlrisk/version.h.in include/mlrisk/version.h @ONLY)

add_library(mlrisk STATIC
    src/rolling.c
    src/sizing.c
    src/vol.c
    src/split.c
    src/linreg.c
    src/version.c
)
add_library(mlrisk::mlrisk ALIAS mlrisk)

# Built as C11; the public headers need only C99, so no standard is exported
target_compile_features(mlrisk PRIVATE c_std_11)
target_compile_options(mlrisk PRIVATE ${MLRISK_COMPILE_OPTIONS})
target_include_directories(mlrisk PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(UNIX)
    target_link_libraries(mlrisk PUBLIC m)
endif()

if(MLRISK_BUILD_TESTS)
    enable_testing()
    add_executable(mlrisk_tests
        tests/test_main.c
        tests/test_rolling.c
        tests/test_sizing.c
        tests/test_vol.c
        tests/test_split.c
        tests/test_linreg.c
        tests/test_fuzz.c
    )
    target_compile_options(mlrisk_tests PRIVATE ${MLRISK_COMPILE_OPTIONS})
    target_link_libraries(mlrisk_tests PRIVATE mlrisk::mlrisk)
    foreach(module rolling vol sizing split linreg fuzz)
        add_test(NAME ${module} COMMAND mlrisk_tests ${module})
    endforeach()
endif()

if(MLRISK_BUILD_EXAMPLES)
    add_executable(vol_target_demo examples/vol_target_demo.c)
    target_compile_options(vol_target_demo PRIVATE ${MLRISK_COMPILE_OPTIONS})
    target_link_libraries(vol_target_demo PRIVATE mlrisk::mlrisk)

endif()

if(MLRISK_BUILD_BENCH)
    add_executable(mlrisk_bench bench/bench.c)
    target_compile_options(mlrisk_bench PRIVATE ${MLRISK_COMPILE_OPTIONS})
    target_link_libraries(mlrisk_bench PRIVATE mlrisk::mlrisk)
endif()

if(MLRISK_INSTALL)
include(CMakePackageConfigHelpers)

install(TARGETS mlrisk EXPORT mlriskTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/mlrisk/version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlrisk)
install(EXPORT mlriskTargets NAMESPACE mlrisk::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlrisk)

configure_package_config_file(cmake/mlriskConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlrisk)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlrisk)

# The .pc file locates the prefix relative to its own directory, whatever
# the depth of CMAKE_INSTALL_LIBDIR (lib, lib64, lib/x86_64-linux-gnu, ...)
file(RELATIVE_PATH MLRISK_PC_TO_PREFIX
     "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
string(REGEX REPLACE "/$" "" MLRISK_PC_TO_PREFIX "${MLRISK_PC_TO_PREFIX}")
configure_file(cmake/mlrisk.pc.in ${CMAKE_CURRENT_BINARY_DIR}/mlrisk.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mlrisk.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()
