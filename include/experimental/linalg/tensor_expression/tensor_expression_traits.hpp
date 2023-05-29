//==================================================================================================
//  File:       tensor_expression_traits.hpp
//
//  Summary:    This header defines tensor expression traits:
//              is_commutative< TE >
//              is_associative< FirstTE, SecondTE >
//              is_distributive< FirstTE, SecondTE >
//              layout_result< Tensor >
//              accessor_result< Tensor >
//              allocator_result< Tensor >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_TENSOR_EXPRESSION_TRAITS_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_TENSOR_EXPRESSION_TRAITS_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

// Is Commutative
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE >
#else
template < class TE, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE > > >
#endif
struct is_commutative : public ::std::false_type { };
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE >
#else
template < class TE, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE > > >
#endif
inline constexpr bool is_commutative_v = is_commutative< TE >::value;

// Is Associative
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
struct is_associative : public ::std::false_type { };
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
inline constexpr bool is_associative_v = is_associative< TE1, TE2 >::value;

// Is Distributive
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
struct is_left_distributive : public ::std::false_type { };
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
inline constexpr bool is_left_distributive_v = is_left_distributive< TE1, TE2 >::value;

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
struct is_right_distributive : public ::std::false_type { };
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::binary_tensor_expression TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
inline constexpr bool is_right_distributive_v = is_right_distributive< TE1, TE2 >::value;
#ifdef LINALG_ENABLE_CONCEPTS
template < class TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
  requires ( LINALG_CONCEPTS::binary_tensor_expression< TE1 > || LINALG_CONCEPTS::unary_tensor_expression< TE1 > )
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< ( LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > || LINALG_CONCEPTS::unary_tensor_expression_v< TE1 > ) && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
struct is_distributive : public ::std::conditional_t< 
#ifdef LINALG_ENABLE_CONCEPTS
                                                      LINALG_CONCEPTS::binary_tensor_expression< TE1 >,
#else
                                                      LINALG_CONCEPTS::binary_tensor_expression_v< TE1 >,
#endif
                                                      ::std::conditional_t< is_commutative_v< TE1, TE2 > && is_left_distributive_v< TE1, TE2 > && is_right_distributive_v< TE1, TE2 >,
                                                                            ::std::true_type,
                                                                            ::std::false_type >,
                                                      ::std::false_type >
{ };
#ifdef LINALG_ENABLE_CONCEPTS
template < class TE1, LINALG_CONCEPTS::binary_tensor_expression TE2 = TE1 >
  requires ( LINALG_CONCEPTS::binary_tensor_expression< TE1 > || LINALG_CONCEPTS::unary_tensor_expression< TE1 > )
#else
template < class TE1, class TE2, typename = ::std::enable_if_t< ( LINALG_CONCEPTS::binary_tensor_expression_v< TE1 > || LINALG_CONCEPTS::unary_tensor_expression_v< TE1 > ) && LINALG_CONCEPTS::binary_tensor_expression_v< TE2 > > >
#endif
inline constexpr bool is_distributive_v = is_distributive< TE1, TE2 >::value;

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_TENSOR_EXPRESSION_TRAITS_HPP
