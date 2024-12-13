cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

string(TOLOWER ${SKBUILD_PROJECT_NAME} PY_PROJECT)

add_subdirectory(${PY_PROJECT})