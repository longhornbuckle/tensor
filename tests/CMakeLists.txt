macro( tensor_add_test name )
  if( TENSOR_TEST_LANGUAGE )
    set_source_files_properties( ${name} PROPERTIES LANGUAGE ${TENSOR_TEST_LANGUAGE} )
  endif()
  add_executable( ${name} ${name}.cpp )
  if ( EXISTS CMAKE_CXX_STANDARD )
    set_property( TARGET ${name} PROPERTY CXX_STANDARD CMAKE_CXX_STANDARD ) # set c++ version
  else()
    set_property( TARGET ${name} PROPERTY CXX_STANDARD 20 ) # set c++ version
  endif()
  set( CMAKE_CXX_STANDARD_REQUIRED YES )
  set( CMAKE_CXX_EXTENSIONS NO )
  target_link_libraries( ${name} tensor gtest_main gtest )
  target_compile_options( ${name}
    PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>>:-fconcepts -Wall -pedantic -Wextra  -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs>
        $<$<OR:$<CXX_COMPILER_ID:Clang>>:-Wall -pedantic -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unused-local-typedefs>
  )
  add_test( ${name} ${name} )
endmacro()

if( TENSOR_USE_SYSTEM_GTEST )
  find_package( GTest CONFIG REQUIRED )
  add_library( gtest_main ALIAS GTest::gtest_main )
else()
  # adapted from https://github.com/google/googletest/blob/master/googletest/README.md
  configure_file( ${PROJECT_SOURCE_DIR}/cmake/googletest/CMakeLists.txt.in googletest-download/CMakeLists.txt )
  execute_process( COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
  )
  if( result )
    message( FATAL_ERROR "CMake step for googletest failed: ${result}" )
  endif()
  execute_process( COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
  )
  if( result )
    message( FATAL_ERROR "Build step for googletest failed: ${result}" )
  endif()

  # Prevent overriding the parent project's compiler/linker
  # settings on Windows
  set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

  # Add googletest directly to our build. This defines
  # the gtest and gtest_main targets.
  add_subdirectory( ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL
  )
endif()

# Add vector / matrix / tensor tests
tensor_add_test( dr_tensor_test )
tensor_add_test( fs_tensor_test )
tensor_add_test( unary_expressions_test )