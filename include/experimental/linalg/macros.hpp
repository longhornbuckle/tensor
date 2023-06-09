//==================================================================================================
//  File:       config.hpp
//
//  Summary:    This header defines macros for defining behavior in the presence of or in the
//              absence of particular feature support
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_MACROS_HPP
#define LINEAR_ALGEBRA_MACROS_HPP

// namespaces
#if ! ( defined( LINALG ) || defined( LINALG_BEGIN ) || defined( LINALG_END ) )
  #define LINALG                 ::std::experimental
  #define LINALG_BEGIN namespace std { namespace experimental {
  #define LINALG_END             } }
#endif

#if ! ( defined( LINALG_DETAIL ) || defined( LINALG_DETAIL_BEGIN ) || defined( LINALG_DETAIL_END ) )
  #define LINALG_DETAIL       LINALG::detail
  #define LINALG_DETAIL_BEGIN LINALG_BEGIN namespace detail {
  #define LINALG_DETAIL_END   LINALG_END }
#endif

#if ! ( defined( LINALG_CONCEPTS ) || defined( LINALG_CONCEPTS_BEGIN ) || defined( LINALG_CONCEPTS_END ) )
  #define LINALG_CONCEPTS       LINALG::concepts
  #define LINALG_CONCEPTS_BEGIN LINALG_BEGIN namespace concepts {
  #define LINALG_CONCEPTS_END   LINALG_END }
#endif

#if ! ( defined( LINALG_EXPRESSIONS ) || defined( LINALG_EXPRESSIONS_BEGIN ) || defined( LINALG_EXPRESSIONS_END ) )
  #define LINALG_EXPRESSIONS       LINALG::expressions
  #define LINALG_EXPRESSIONS_BEGIN LINALG_BEGIN namespace expressions {
  #define LINALG_EXPRESSIONS_END   LINALG_END }
#endif

#if ! ( defined( LINALG_EXPRESSIONS_DETAIL ) || defined( LINALG_EXPRESSIONS_DETAIL_BEGIN ) || defined( LINALG_EXPRESSIONS_DETAIL_END ) )
  #define LINALG_EXPRESSIONS_DETAIL       LINALG_EXPRESSIONS::detail
  #define LINALG_EXPRESSIONS_DETAIL_BEGIN LINALG_EXPRESSIONS_BEGIN namespace detail {
  #define LINALG_EXPRESSIONS_DETAIL_END   LINALG_EXPRESSIONS_END }
#endif

// Force compiler to inline function
#ifndef LINALG_FORCE_INLINE_FUNCTION
#  ifdef LINALG_COMPILER_MSVC
#    define LINALG_FORCE_INLINE_FUNCTION __forceinline
#  else
#    define LINALG_FORCE_INLINE_FUNCTION __attribute__((always_inline))
#  endif
#endif

//- C++17 related macros

// Support for concepts
// Note, clang does not support non-inline concept function definitions until 16.
#ifndef LINALG_ENABLE_CONCEPTS
#  if defined( __cpp_lib_concepts ) && ( ( LINALG_COMPILER_GNU >= 10 ) || ( LINALG_COMPILER_CLANG >= 16 ) )
#    define LINALG_ENABLE_CONCEPTS
#  endif
#endif

// Support for no throw convertible
#ifndef LINALG_NO_THROW_CONVERTIBLE
#  if defined( __cpp_lib_is_nothrow_convertible ) && LINALG_HAS_CXX_20
#    define LINALG_NO_THROW_CONVERTIBLE
#  endif
#endif

// Constexpr destructor disabled for C++17
#ifndef LINALG_CONSTEXPR_DESTRUCTOR
#  if LINALG_HAS_CXX_20
#    define LINALG_CONSTEXPR_DESTRUCTOR constexpr
#  else
#    define LINALG_CONSTEXPR_DESTRUCTOR
#  endif
#endif

// Lambda expressions may not appear in an unevaluated operand in C++17
#ifndef LINALG_UNEVALUATED_LAMBDA
#  if LINALG_HAS_CXX_20
#    define LINALG_UNEVALUATED_LAMBDA
#  endif
#endif

// Likely not supported until C++20
#ifndef LINALG_LIKELY
#  if LINALG_HAS_CXX_20
#    define LINALG_LIKELY [[likely]]
#  else
#    define LINALG_LIKELY
#  endif
#endif

// Unlikely not supported until C++20
#ifndef LINALG_UNLIKELY
#  if LINALG_HAS_CXX_20
#    define LINALG_UNLIKELY [[unlikely]]
#  else
#    define LINALG_UNLIKELY
#  endif
#endif

//- C++20 related macros

// Define if STL execution policies are supported.
#ifndef LINALG_EXECUTION_POLICY
#  if defined( __cpp_lib_execution ) && ( ( LINALG_COMPILER_GNU >= 9 ) || ( LINALG_COMPILER_MSVC >= 1914 ) )
#    define LINALG_EXECTUION_POLICY 1
#  else
#    define LINALG_EXECUTION_POLICY 0
#  endif
#endif

// Define execution::seq if available.
#ifndef LINALG_EXECUTION_SEQ
#  if LINALG_EXECTUION_POLICY
#    define LINALG_EXECUTION_SEQ ::std::execution::seq
#  else
#    define LINALG_EXECUTION_SEQ 0
#  endif
#endif

// Define execution::unseq if available.
// If not, then just use execution::seq instead.
#ifndef LINALG_EXECUTION_UNSEQ
#  if LINALG_EXECTUION_POLICY && ( __cpp_lib_execution >= 201902L ) && ( ( LINALG_COMPILER_GNU >= 9 ) || ( LINALG_COMPILER_MSVC >= 1928 ) )
#    define LINALG_EXECUTION_UNSEQ ::std::execution::unseq
#  else
#    define LINALG_EXECUTION_UNSEQ LINALG_EXECUTION_SEQ
#  endif
#endif

#ifndef LINALG_ENABLE_RANGES
#  if ( __cpp_lib_ranges >= 201911L ) && ( ( LINALG_COMPILER_GNU >= 10 ) || ( LINALG_COMPILER_CLANG >= 15 ) || ( LINALG_COMPILER_MSVC >= 1929 ) )
#    define LINALG_ENABLE_RANGES
#  endif
#endif

//- C++23 related macros

// Determine if operator[](...) is allowable
#ifndef LINALG_USE_BRACKET_OPERATOR
#  if defined( __cpp_multidimensional_subscript ) && \
      ( ( defined( LINALG_COMPILER_GNU ) && ( LINALG_COMPILER_GNU >= 12 ) ) || \
        ( defined( LINALG_COMPILER_CLANG ) && ( LINALG_COMPILER_CLANG >= 15 ) ) )
#    define LINALG_USE_BRACKET_OPERATOR 1
#  else
#    define LINALG_USE_BRACKET_OPERATOR 0
#  endif
#endif

// If operator[](...) is not allowable, then use operator()(...)
#ifndef LINALG_USE_PAREN_OPERATOR
#  if !LINALG_USE_BRACKET_OPERATOR
#    define LINALG_USE_PAREN_OPERATOR 1
#  else
#    define LINALG_USE_PAREN_OPERATOR 0
#  endif
#endif

// Check if ranges::to available
#ifndef LINALG_RANGES_TO_CONTAINER
#  if defined( LINALG_ENABLE_RANGES ) && ( __cpp_lib_ranges_to_container >= 202202L )
#    define LINALG_RANGES_TO_CONTAINER
#  endif
#endif

#endif  //- LINEAR_ALGEBRA_MACROS_HPP
