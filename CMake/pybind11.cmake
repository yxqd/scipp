# Download and unpack pybind11 at configure time
configure_file(${CMAKE_SOURCE_DIR}/CMake/pybind11.in
               ${CMAKE_BINARY_DIR}/pybind11-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pybind11-download )

execute_process(COMMAND ${CMAKE_COMMAND} --build .
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pybind11-download )

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
add_subdirectory(${CMAKE_BINARY_DIR}/pybind11-src
                 ${CMAKE_BINARY_DIR}/pybind11-src)
