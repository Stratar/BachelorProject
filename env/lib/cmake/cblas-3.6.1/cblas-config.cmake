# Compute locations from <prefix>/lib/cmake/lapacke-<v>/<self>.cmake
get_filename_component(_CBLAS_SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_CBLAS_PREFIX "${_CBLAS_SELF_DIR}" PATH)
get_filename_component(_CBLAS_PREFIX "${_CBLAS_PREFIX}" PATH)
get_filename_component(_CBLAS_PREFIX "${_CBLAS_PREFIX}" PATH)

# Load the LAPACK package with which we were built.
set(LAPACK_DIR "${_CBLAS_PREFIX}/lib/cmake/lapack-3.6.1")
find_package(LAPACK NO_MODULE)

# Load lapacke targets from the install tree.
if(NOT TARGET cblas)
  include(${_CBLAS_SELF_DIR}/cblas-targets.cmake)
endif()

# Report lapacke header search locations.
set(CBLAS_INCLUDE_DIRS ${_CBLAS_PREFIX}/include)

# Report lapacke libraries.
set(CBLAS_LIBRARIES cblas)

unset(_CBLAS_PREFIX)
unset(_CBLAS_SELF_DIR)
