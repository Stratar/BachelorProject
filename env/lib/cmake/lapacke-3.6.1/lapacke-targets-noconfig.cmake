#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lapacke" for configuration ""
set_property(TARGET lapacke APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lapacke PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "lapack;blas"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liblapacke.so"
  IMPORTED_SONAME_NOCONFIG "liblapacke.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS lapacke )
list(APPEND _IMPORT_CHECK_FILES_FOR_lapacke "${_IMPORT_PREFIX}/lib/liblapacke.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
