# Builds the mlrisk C sources into one shared library that the Python package
# loads with ctypes. The C is the same code the CMake and CI builds use; only
# the linkage differs, so the Python package cannot drift from the library.
cmake_minimum_required(VERSION 3.21)

# One source of truth for the version: the root CMakeLists
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" MLRISK_ROOT_CMAKE)
string(REGEX MATCH "project\\(mlrisk VERSION ([0-9]+)\\.([0-9]+)\\.([0-9]+)"
       MLRISK_VERSION_MATCH "${MLRISK_ROOT_CMAKE}")
if(NOT MLRISK_VERSION_MATCH)
    message(FATAL_ERROR "could not read the project version from ../CMakeLists.txt")
endif()

project(walkforward VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3} LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(MLRISK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
file(GLOB MLRISK_SOURCES CONFIGURE_DEPENDS "${MLRISK_ROOT}/src/*.c")
configure_file("${MLRISK_ROOT}/include/mlrisk/version.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/mlrisk/version.h" @ONLY)

add_library(walkforward_native SHARED ${MLRISK_SOURCES})
target_include_directories(walkforward_native PRIVATE
    "${MLRISK_ROOT}/include" "${CMAKE_CURRENT_BINARY_DIR}/include")
if(MSVC)
    target_compile_options(walkforward_native PRIVATE /W4 /fp:precise /fp:contract-)
else()
    target_compile_options(walkforward_native PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(walkforward_native PRIVATE m)
endif()
set_target_properties(walkforward_native PROPERTIES
    PREFIX ""                                # walkforward_native.so / .dylib / .dll
    C_VISIBILITY_PRESET default
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

install(TARGETS walkforward_native DESTINATION walkforward)
