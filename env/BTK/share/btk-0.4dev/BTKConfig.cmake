#---------------------------------------------------------------------------
#
# BTKConfig.cmake - BTK CMake configuration file for external projects.
#
# This file is configured by BTK and used by the UseBTK.cmake module
# to load BTK's settings for an external project.

# Compute the installation prefix from this BTKConfig.cmake file location.
GET_FILENAME_COMPONENT(BTK_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
GET_FILENAME_COMPONENT(BTK_INSTALL_PREFIX "${BTK_INSTALL_PREFIX}" PATH)
GET_FILENAME_COMPONENT(BTK_INSTALL_PREFIX "${BTK_INSTALL_PREFIX}" PATH)

# The BTK include file directories.
SET(BTK_INCLUDE_DIRS "${BTK_INSTALL_PREFIX}/include/btk-0.4dev;${BTK_INSTALL_PREFIX}/include/btk-0.4dev/BasicFilters;${BTK_INSTALL_PREFIX}/include/btk-0.4dev/Common;${BTK_INSTALL_PREFIX}/include/btk-0.4dev/IO;${BTK_INSTALL_PREFIX}/include/btk-0.4dev/Utilities;${BTK_INSTALL_PREFIX}/include/btk-0.4dev/Utilities/eigen3")

# The BTK library directories.
SET(BTK_LIBRARY_DIRS "${BTK_INSTALL_PREFIX}/lib/btk-0.4dev")

# The C and C++ flags added by BTK to the cmake-configured flags.
SET(BTK_REQUIRED_C_FLAGS "")
SET(BTK_REQUIRED_CXX_FLAGS "")
SET(BTK_REQUIRED_LINK_FLAGS "")

# The BTK version number
SET(BTK_VERSION_MAJOR "0")
SET(BTK_VERSION_MINOR "4dev")
SET(BTK_VERSION_PATCH "0")

# The location of the UseBTK.cmake file.
SET(BTK_USE_FILE "${BTK_INSTALL_PREFIX}/share/btk-0.4dev/UseBTK.cmake")

# The build settings file.
SET(BTK_BUILD_SETTINGS_FILE "${BTK_INSTALL_PREFIX}/share/btk-0.4dev/BTKBuildSettings.cmake")

# The library dependencies file.
SET(BTK_LIBRARY_DEPENDS_FILE "${BTK_INSTALL_PREFIX}/share/btk-0.4dev/BTKLibraryDepends.cmake")

# Whether BTK was built with shared libraries.
SET(BTK_BUILD_SHARED "ON")

# Whether BTK was built with the Matlab wrapping support.
SET(BTK_WRAP_MATLAB "OFF")

# Whether BTK was built with the Octave wrapping support.
SET(BTK_WRAP_OCTAVE "OFF")

# Whether BTK was built with the Python wrapping support.
SET(BTK_WRAP_PYTHON "OFF")

# Whether BTK was built with the Scilab wrapping support.
SET(BTK_WRAP_SCILAB "")

# A list of all libraries for BTK.
# Those listed here should automatically pull in their dependencies.
SET(BTK_LIBRARIES BTKCommon;BTKBasicFilters;BTKIO)

# The BTK library dependencies.
IF(NOT BTK_NO_LIBRARY_DEPENDS AND EXISTS "${BTK_LIBRARY_DEPENDS_FILE}")
  INCLUDE("${BTK_LIBRARY_DEPENDS_FILE}")
ENDIF(NOT BTK_NO_LIBRARY_DEPENDS AND EXISTS "${BTK_LIBRARY_DEPENDS_FILE}")
