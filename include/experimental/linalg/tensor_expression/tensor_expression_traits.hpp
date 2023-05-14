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

namespace std
{
namespace experimental
{

// Is Commutative
template < binary_tensor_expression TE >
struct is_commutative : public false_type;
template < binary_tensor_expression TE >
[[nodiscard]] inline constexpr bool is_commutative_v = is_commutative< TE >::value;

// Is Associative
template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
struct is_associative : public false_type;
template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
[[nodiscard]] inline constexpr bool is_associative_v = is_associative< TE >::value;

// Is Distributive
template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
struct is_distributive< TE1, TE2 > : public false_type;
template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
[[nodiscard]] inline constexpr bool is_distributive_v = is_commutative< TE1, TE2 > && is_distributive< TE1, TE2 >::value;

template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
struct is_left_distributive< TE1, TE2 > : public false_type;
template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
[[nodiscard]] inline constexpr bool is_left_distributive_v = is_distributive_v< TE1, TE2 > || is_left_distributive< TE1, TE2 >::value;

template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
struct is_right_distributive< TE1, TE2 > : public false_type;
template < binary_tensor_expression TE1, binary_tensor_expression TE2 = TE1 >
[[nodiscard]] inline constexpr bool is_right_distributive_v = is_distributive_v< TE1, TE2 > || is_right_distributive< TE1, TE2 >::value;

template < unary_tensor_expression TE1, binary_tensor_expression TE2 >
struct is_distributive< TE1, TE2 > : public false_type;
template < unary_tensor_expression TE1, binary_tensor_expression TE2 >
[[nodiscard]] inline constexpr bool is_distributive_v = is_distributive< TE1, TE2 >::value;

// Layout result defines the resultant layout of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct layout_result;

template < readable_tensor Tensor >
struct layout_result< negate_tensor_expression< Tensor > >
{
  using type = typename Tensor::layout_type;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< negate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < readable_tensor Tensor >
struct layout_result< transpose_tensor_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_stride > )
{
  using type = default_layout;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< transpose_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < readable_tensor Tensor >
struct layout_result< conjugate_tensor_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_stride > )
{
  using type = default_layout;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< conjugate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < readable_tensor FirstTensor, readable_tensor SecondTensor >
struct layout_result< add_tensor_expression< FirstTensor, SecondTensor > >
  requires ( ( ::std::is_same_v< typename FirstTensor::layout_type, ::std::experimental::layout_right > ||
               ::std::is_same_v< typename FirstTensor::layout_type, ::std::experimental::layout_left > ||
               ::std::is_same_v< typename FirstTensor::layout_type, ::std::experimental::layout_stride > ) &&
             ( ::std::is_same_v< typename SecondTensor::layout_type, ::std::experimental::layout_right > ||
               ::std::is_same_v< typename SecondTensor::layout_type, ::std::experimental::layout_left > ||
               ::std::is_same_v< typename SecondTensor::layout_type, ::std::experimental::layout_stride > ) )
{
  using type = ::std::conditional_t< ::std::is_same_v< typename FirstTensor::layout_type, typename SecondTensor::layout_type > &&
                                       ! ::std::is_same_v< typename FirstTensor::layout_type, ::std::experimental::layout_stride >,
                                     typename FirstTensor::layout_type,
                                     default_layout >;
};

template < unevaluated_tensor_expression FirstTensor, readable_tensor SecondTensor >
struct layout_result< add_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< add_tensor_expression< decltype( ::std::declval< FirstTensor >().operator auto() ), SecondTensor > >::type;
};

template < readable_tensor FirstTensor, unevaluated_tensor_expression SecondTensor >
struct layout_result< add_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< add_tensor_expression< FirstTensor, decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
};

template < unevaluated_tensor_expression FirstTensor, unevaluated_tensor_expression SecondTensor >
struct layout_result< add_tensor_expression< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< add_tensor_expression< decltype( ::std::declval< FirstTensor >().operator auto() ), decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
};

template < tensor_expression Tensor >
using layout_result_t = typename layout_result< Tensor >::type;


// Accessor result defines the resultant accessor of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct accessor_result;

template < class T >
class is_default_accessor< T > : public std::false_type { }

template < class T >
class is_default_accessor< ::std::experimental::default_accessor< T > > : public std::true_type { }

template < class T >
[[nodiscard]] inline constexpr bool is_default_accessor_v = is_default_accessor< T >::value;

template < readable_tensor Tensor >
struct accessor_result< negate_tensor_expression< Tensor > >
{
  using type = typename Tensor::accessor_type;
};

template < unevaluated_tensor_expression Tensor >
struct accessor_result< negate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < readable_tensor Tensor >
struct accessor_result< transpose_tensor_expression< Tensor > >
{
  using type = typename Tensor::accessor_type;
};

template < unevaluated_tensor_expression Tensor >
struct accessor_result< conjugate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < readable_tensor FirstTensor, readable_tensor SecondTensor >
struct accessor_result< add_tensor_expression< FirstTensor, SecondTensor > >
  requires ( is_default_accessor_v< typename FirstTensor::accessor_type > &&
             is_default_accessor_v< typename SecondTensor::accessor_type > )
{
  using type = ::std::experimental::default_accessor< decltype( ::std::declval< typename FirstTensor::value_type >() + decltype( typename SecondTensor::value_type >() ) >;
};

template < class Tensor >
using accessor_result_t = typename accessor_result< Tensor >::type;


// Allocator result defines the resultant allocator of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct allocator_result
{
  using type = ::std::allocator< typename Tensor::value_type >;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& ) noexcept { return type(); }
};

template < dynamic_tensor Tensor >
struct allocator_result< Tensor >
{
  using type = typename Tensor::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& t ) noexcept
  {
    if constexpr ( ! ::std::is_rvalue_reference_v<Tensor> )
    {
      return ::std::allocator_traits< typename Tensor::allocator_type >::select_on_container_copy_construction( t.get_allocator() );
    }
    else
    {
      return t.get_allocator();
    }
  }
};

template < unary_tensor_expression Tensor >
struct allocator_result< Tensor >
{
  using type = typename allocator_result< decltype( ::std::declval<Tensor>.underlying() ) >::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& t ) noexcept { return allocator_result< decltype( ::std::declval<Tensor>.underlying() ) >::get_allocator( t.underlying(); ) }
};

template < binary_tensor_expression Tensor >
struct allocator_result< Tensor >
{
  using type = typename allocator_result< ::std::conditional_t< dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.first() ) > > ||
                                                                  !dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.second() ) > >,
                                                                decltype( ::std::declval<Tensor>.first() ),
                                                                decltype( ::std::declval<Tensor>.second() ) >::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& t ) noexcept
  {
    if constexpr ( dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.first() ) > > ||
                   !dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.second() ) > )
    {
      return allocator_result< decltype( ::std::declval<Tensor>.first() ) >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< decltype( ::std::declval<Tensor>.second() ) >::get_allocator( t.second() );
    }
  }
};

template < class Tensor >
using allocator_result_t = typename allocator_result< Tensor >::type;

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_TENSOR_EXPRESSION_TRAITS_HPP
