#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "blas" for configuration ""
set_property(TARGET blas APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(blas PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libblas.so.3.6.1"
  IMPORTED_SONAME_NOCONFIG "libblas.so.3"
  )

list(APPEND _IMPORT_CHECK_TARGETS blas )
list(APPEND _IMPORT_CHECK_FILES_FOR_blas "${_IMPORT_PREFIX}/lib/libblas.so.3.6.1" )

# Import target "lapack" for configuration ""
set_property(TARGET lapack APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lapack PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "blas"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liblapack.so.3.6.1"
  IMPORTED_SONAME_NOCONFIG "liblapack.so.3"
  )

list(APPEND _IMPORT_CHECK_TARGETS lapack )
list(APPEND _IMPORT_CHECK_FILES_FOR_lapack "${_IMPORT_PREFIX}/lib/liblapack.so.3.6.1" )

# Import target "tmglib" for configuration ""
set_property(TARGET tmglib APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(tmglib PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "lapack"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtmglib.so"
  IMPORTED_SONAME_NOCONFIG "libtmglib.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS tmglib )
list(APPEND _IMPORT_CHECK_FILES_FOR_tmglib "${_IMPORT_PREFIX}/lib/libtmglib.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
