//==================================================================================================
//  File:       tensor_expression_traits.hpp
//
//  Summary:    This header defines tensor expression traits:
//              rebind_accessor< Accessor, T >
//              accessor_result< Tensor >
//              allocator_result< Tensor >
//              layout_result< Tensor >
//              is_alias_assignable< Tensor >
//
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

//-------------------
//  Rebind Accessor
//-------------------

// Helper rebinds the accessor to the new type
template < class Accessor, class T = ::std::remove_cv_t< typename Accessor::element_type > >
struct rebind_accessor;

template < class T, class U >
struct rebind_accessor< ::std::default_accessor< T >, U >
{
  using type = ::std::default_accessor< U >;
};

template < class T, class U >
using rebind_accessor_t = typename rebind_accessor< T, U >::type;

//-------------------
//  Accessor Result
//-------------------

// Accessor result defines the resultant accessor of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct accessor_result;

template < class Tensor >
using accessor_result_t = typename accessor_result< Tensor >::type;

//--------------------
//  Allocator Result
//--------------------

// Allocator result defines the resultant allocator of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct allocator_result
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct default_helper
  {
    using type = ::std::allocator< typename ::std::remove_reference_t< T >::value_type >;
    [[nodiscard]] static inline constexpr type get_allocator( const T&& ) noexcept { return type(); }
  };
  template < class T >
  struct dynamic_helper
  {
    using type = typename ::std::remove_reference_t< T >::allocator_type;
    [[nodiscard]] static inline constexpr type get_allocator( const T&& t ) noexcept
    {
      if constexpr ( ! ::std::is_rvalue_reference_v< T > )
      {
        return ::std::allocator_traits< typename ::std::remove_reference_t< T >::allocator_type >::select_on_container_copy_construction( t.get_allocator() );
      }
      else
      {
        return t.get_allocator();
      }
    }
  };
public:
  using type = typename ::std::conditional_t< 
                                              #ifdef LINALG_ENABLE_CONCEPTS
                                              LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< Tensor > >,
                                              #else
                                              LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Tensor > >,
                                              #endif
                                              dynamic_helper< Tensor >,
                                              ::std::conditional_t< 
                                                                    #ifdef LINALG_ENABLE_CONCEPTS
                                                                    LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >,
                                                                    #else
                                                                    LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                    #endif
                                                                    default_helper< Tensor >,
                                                                    invalid_helper< Tensor > > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const Tensor&& t ) noexcept
  {
    #ifdef LINALG_ENABLE_CONCEPTS
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< Tensor > > )
    #else
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Tensor > > )
    #endif
    {
      return dynamic_helper< Tensor >::get_allocator( ::std::forward< const Tensor >( t ) );
    }
    #ifdef LINALG_ENABLE_CONCEPTS
    else if constexpr ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > > )
    #else
    else if constexpr ( LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > )
    #endif
    {
      return default_helper< Tensor >::get_allocator( ::std::forward< const Tensor >( t ) );
    }
    else
    {
      return invalid_helper< Tensor >::get_allocator( ::std::forward< const Tensor >( t ) );
    }
  }
};

template < class Tensor >
using allocator_result_t = typename allocator_result< Tensor >::type;

//-----------------
//  Layout Result
//-----------------

// Layout result defines the resultant layout of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct layout_result;

template < class Tensor >
using layout_result_t = typename layout_result< Tensor >::type;

//--------------------
//  Alias Assignable
//--------------------
template < class Tensor >
struct is_alias_assignable : public ::std::false_type { };

template < class Tensor >
inline constexpr bool is_alias_assignable_v = is_alias_assignable< Tensor >::value;







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
