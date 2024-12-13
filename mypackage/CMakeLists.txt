
# Build the PyBind11 module from the .cpp source file
# Module name should fit the name in the PyBind11 module (in .cpp)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(CURRENT_MODULE "tissue")
set(SRC_FILES
    "src_${CURRENT_MODULE}/tissue.cpp"
)

nanobind_add_module("${CURRENT_MODULE}" ${SRC_FILES})
install(TARGETS "${CURRENT_MODULE}" LIBRARY DESTINATION "${PY_PROJECT}")

