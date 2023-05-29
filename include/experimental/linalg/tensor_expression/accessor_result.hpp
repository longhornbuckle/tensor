//==================================================================================================
//  File:       accessor_result.hpp
//
//  Summary:    This header defines tensor expression traits:
//              accessor_result< Tensor >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_ACCESSOR_RESULT_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_ACCESSOR_RESULT_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

// Helper is true if type is a std::default_accessor
template < class T >
class is_default_accessor : public ::std::false_type { };

template < class T >
class is_default_accessor< ::std::default_accessor< T > > : public ::std::true_type { };

template < class T >
inline constexpr bool is_default_accessor_v = is_default_accessor< T >::value;

// Helper removes constness from returned accessor's element type
template < class T >
struct remove_accessor_constness;

template < class T >
struct remove_accessor_constness< ::std::default_accessor< T > >
{
  using type = ::std::default_accessor< ::std::remove_cv_t< T > >;
};

// Accessor result defines the resultant accessor of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct accessor_result;

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = typename remove_accessor_constness< typename ::std::remove_reference_t< Tensor >::accessor_type >::type;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = typename remove_accessor_constness< typename ::std::remove_reference_t< Tensor >::accessor_type >::type;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose > >
{
  using type = typename remove_accessor_constness< typename ::std::remove_reference_t< Tensor >::accessor_type >::type;
};

template < class Tensor >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};
#else
template < class Tensor,
           class Enable >
struct accessor_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = typename ::std::conditional_t< is_default_accessor_v< typename ::std::remove_reference_t< T >::accessor_type >,
                                                remove_accessor_constness< typename ::std::remove_reference_t< T >::accessor_type >,
                                                invalid_helper< T > >::type;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< T >().operator auto() )::accessor_type;
  };
public:
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                              readable_helper< Tensor >,
                                              ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                    unevaluated_helper< Tensor >,
                                                                    invalid_helper< Tensor > > >::type;
};

template < class Tensor,
           class Transpose,
           class Enable >
struct accessor_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = typename ::std::conditional_t< is_default_accessor_v< typename ::std::remove_reference_t< T >::accessor_type >,
                                                remove_accessor_constness< typename ::std::remove_reference_t< T >::accessor_type >,
                                                invalid_helper< T > >::type;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< T >().operator auto() )::accessor_type;
  };
public:
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                              readable_helper< Tensor >,
                                              ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                    unevaluated_helper< Tensor >,
                                                                    invalid_helper< Tensor > > >::type;
};

template < class Tensor,
           class Transpose,
           class Enable >
struct accessor_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = typename ::std::conditional_t< is_default_accessor_v< typename ::std::remove_reference_t< T >::accessor_type >,
                                                remove_accessor_constness< typename ::std::remove_reference_t< T >::accessor_type >,
                                                invalid_helper< T > >::type;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< T >().operator auto() )::accessor_type;
  };
public:
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                              readable_helper< Tensor >,
                                              ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                    unevaluated_helper< Tensor >,
                                                                    invalid_helper< Tensor > > >::type;
};
#endif

// #ifdef LINALG_ENABLE_CONCEPTS
// template < template < class, class > class BTE,
//            class                           FirstTensor,
//            class                           SecondTensor >
//   requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
//              LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondTensor > > &&
//              is_default_accessor_v< typename ::std::remove_reference_t< FirstTensor >::accessor_type > &&
//              is_default_accessor_v< typename ::std::remove_reference_t< SecondTensor >::accessor_type > )
// struct accessor_result< BTE< FirstTensor, SecondTensor > >
// {
//   using type = ::std::default_accessor< typename ::std::remove_reference_t< BTE< FirstTensor, SecondTensor > >::value_type >;
// };
// #else
// template < template < class, class, class Enable > class BTE,
//            class                                         FirstTensor,
//            class                                         SecondTensor,
//            class                                         Enable >
// struct accessor_result< BTE< FirstTensor, SecondTensor, Enable > >
// {
// private:
//   template < class T, class U >
//   struct invalid_helper;
//   template < class T, class U >
//   struct default_helper
//   {
//     template < class V >
//     struct helper
//     {
//       using type = ::std::default_accessor< ::std::remove_reference_t< V > >;
//     };
    
//     using type = typename ::std::conditional_t< ( is_default_accessor_v< typename ::std::remove_reference_t< T >::accessor_type > &&
//                                                   is_default_accessor_v< typename ::std::remove_reference_t< U >::accessor_type > ),
//                                                 helper< BTE< FirstTensor, SecondTensor, Enable > >,
//                                                 invalid_helper< T, U > >::type;
//   };
// public:
//   using type = typename ::std::conditional_t< ( LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< FirstTensor > > &&
//                                                 LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< SecondTensor > > ),
//                                               default_helper< FirstTensor, SecondTensor >,
//                                               invalid_helper< FirstTensor, SecondTensor > >::type;
// };
// #endif

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
using accessor_result_t = typename accessor_result< Tensor >::type;

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_ACCESSOR_RESULT_HPP
