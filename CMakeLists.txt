cmake_minimum_required(VERSION 3.21)
project(mlrisk VERSION 3.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(MLRISK_WERROR "Treat warnings as errors" ${PROJECT_IS_TOP_LEVEL})
option(MLRISK_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(MLRISK_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})

if(MSVC)
    set(MLRISK_COMPILE_OPTIONS /W4 /fp:precise $<$<BOOL:${MLRISK_WERROR}>:/WX>)
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
)
add_library(mlrisk::mlrisk ALIAS mlrisk)

target_compile_features(mlrisk PUBLIC c_std_11)
target_compile_options(mlrisk PRIVATE ${MLRISK_COMPILE_OPTIONS})
target_include_directories(mlrisk PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
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
    )
    target_compile_options(mlrisk_tests PRIVATE ${MLRISK_COMPILE_OPTIONS})
    target_link_libraries(mlrisk_tests PRIVATE mlrisk::mlrisk)
    foreach(module rolling vol sizing split linreg)
        add_test(NAME ${module} COMMAND mlrisk_tests ${module})
    endforeach()
endif()

if(MLRISK_BUILD_EXAMPLES)
    add_executable(vol_target_demo examples/vol_target_demo.c)
    target_compile_options(vol_target_demo PRIVATE ${MLRISK_COMPILE_OPTIONS})
    target_link_libraries(vol_target_demo PRIVATE mlrisk::mlrisk)
endif()

include(GNUInstallDirs)
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

configure_file(cmake/mlrisk.pc.in ${CMAKE_CURRENT_BINARY_DIR}/mlrisk.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mlrisk.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
