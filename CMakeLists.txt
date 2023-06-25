cmake_minimum_required( VERSION 3.15 )
project( TENSOR VERSION 0.1.0 LANGUAGES CXX )

include( GNUInstallDirs )

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

add_library( tensor INTERFACE )

target_include_directories( tensor INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/externals/mdspan/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

################################################################################

install( TARGETS tensor EXPORT tensorTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install( EXPORT tensorTargets
    FILE tensorTargets.cmake
    NAMESPACE std::experimental::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor
)

export( TARGETS tensor
    NAMESPACE std::experimental::
    FILE tensorTargets.cmake
)

install( DIRECTORY include/experimental DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

include( CMakePackageConfigHelpers )
configure_package_config_file( cmake/tensorConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/tensorConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor
)
write_basic_package_version_file( ${CMAKE_CURRENT_BINARY_DIR}/tensorConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT
)

install( FILES ${CMAKE_CURRENT_BINARY_DIR}/tensorConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/tensorConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor
)

################################################################################

enable_testing()
add_subdirectory(tests)