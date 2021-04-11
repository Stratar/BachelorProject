# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.4)
  # Information for CMake 2.6 and above.
  set("BTKBasicFilters_LIB_DEPENDS" "general;BTKCommon;")
  set("BTKIO_LIB_DEPENDS" "general;BTKCommon;")
else()
  # Information for CMake 2.4 and lower.
  set("BTKBasicFilters_LIB_DEPENDS" "BTKCommon;")
  set("BTKIO_LIB_DEPENDS" "BTKCommon;")
endif()
