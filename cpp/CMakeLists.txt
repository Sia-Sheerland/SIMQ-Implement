cmake_minimum_required(VERSION 3.14)
project(simq_main_experiment CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

set(HNSWLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hnswlib/hnswlib")

add_executable(simq_index_builder simq_index_builder.cpp)
target_include_directories(simq_index_builder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HNSWLIB_DIR}/..
)
target_compile_options(simq_index_builder PRIVATE -fopenmp)
target_link_libraries(simq_index_builder PRIVATE pthread -fopenmp)

add_executable(simq_query simq_query.cpp)
target_include_directories(simq_query PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HNSWLIB_DIR}/..
)
target_link_libraries(simq_query PRIVATE pthread aio)
