cmake_minimum_required(VERSION 3.15)
project(NBodyOrbitalPropagator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody_propagator src/main.cpp)

# Create output directory at build time
add_custom_command(TARGET nbody_propagator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/output"
)
