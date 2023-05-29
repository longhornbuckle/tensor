//==================================================================================================
//  File:       layout_result.hpp
//
//  Summary:    This header defines tensor expression traits:
//              layout_result< Tensor >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_LAYOUT_RESULT_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_LAYOUT_RESULT_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

// Layout result defines the resultant layout of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct layout_result;

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride >,
                                     default_layout,
                                     typename ::std::remove_reference_t< Tensor >::layout_type >;
};

template < class Tensor >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct layout_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};
#else
template < class Tensor, class Enable >
struct layout_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable > >
{
private :
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< T >::layout_type, ::std::layout_stride >,
                                       default_layout,
                                       typename ::std::remove_reference_t< T >::layout_type >;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< ::std::remove_reference< T > >().operator auto() )::layout_type;
  };
public :
  using type = typename ::std::conditional_t< ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
                                                ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
                                                ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ),
                                              ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< Tensor >,
                                                                    readable_helper< Tensor >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< Tensor >,
                                                                                          unevaluated_helper< Tensor >,
                                                                                          readable_helper< Tensor > > >,
                                              invalid_helper< Tensor > >::type;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = default_layout;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};
#else
template < class Tensor, class Transpose, class Enable >
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable > >
{
private :
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = default_layout;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
  };
public :
  using type = typename ::std::conditional_t< ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
                                                ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
                                                ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ),
                                              ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                                                    readable_helper< Tensor >,
                                                                   ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                                         unevaluated_helper< Tensor >,
                                                                                         invalid_helper< Tensor > > >,
                                     invalid_helper< Tensor > >::type;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose > >
{
  using type = default_layout;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct layout_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};
#else
template < class Tensor, class Transpose, class Enable >
struct layout_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose, Enable > >
{
private :
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = default_layout;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
  };
public :
  using type = ::std::conditional_t< ( ::std::is_same_v< typename Tensor::layout_type, ::std::layout_right > ||
                                       ::std::is_same_v< typename Tensor::layout_type, ::std::layout_left > ||
                                       ::std::is_same_v< typename Tensor::layout_type, ::std::layout_stride > ),
                                     ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< Tensor >,
                                                           typename readable_helper< Tensor >::type,
                                                           ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< Tensor >,
                                                                                 typename unevaluated_helper< Tensor >::type,
                                                                                 typename invalid_helper< Tensor >::type > >,
                                     typename invalid_helper< Tensor >::type >;
};
#endif

// #ifdef LINALG_ENABLE_CONCEPTS
// template < template < class, class > class  BTE,
//            class                            FirstTensor,
//            class                            SecondTensor >
//   requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstTensor > > &&
//              LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondTensor > > &&
//              ( ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_right > ||
//                ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_left > ||
//                ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_stride > ) &&
//              ( ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_right > ||
//                ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_left > ||
//                ::std::is_same_v< typename ::std::remove_reference_t< SecondTensor >::layout_type, ::std::layout_stride > ) )
// struct layout_result< BTE< FirstTensor, SecondTensor > >
// {
//   using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, typename ::std::remove_reference_t< SecondTensor >::layout_type > &&
//                                        ! ::std::is_same_v< typename ::std::remove_reference_t< FirstTensor >::layout_type, ::std::layout_stride >,
//                                      typename ::std::remove_reference_t< FirstTensor >::layout_type,
//                                      default_layout >;
// };

// template < template < class, class > class BTE,
//            class                           FirstTensor,
//            class                           SecondTensor >
//   requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
//              LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondTensor > > )
// struct layout_result< BTE< FirstTensor, SecondTensor > >
// {
//   using type = typename layout_result< ::std::remove_reference_t< BTE< decltype( ::std::declval< FirstTensor >().operator auto() ), SecondTensor > > >::type;
// };

// template < template < class, class > class BTE,
//            class                           FirstTensor,
//            class                           SecondTensor >
//   requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstTensor > > &&
//              LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondTensor > > )
// struct layout_result< BTE< FirstTensor, SecondTensor > >
// {
//   using type = typename layout_result< ::std::remove_reference_t< BTE< FirstTensor, decltype( ::std::declval< SecondTensor >().operator auto() ) > > >::type;
// };

// template < template < class, class > class BTE,
//            class                           FirstTensor,
//            class                           SecondTensor >
//   requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
//              LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondTensor > > )
// struct layout_result< BTE< FirstTensor, SecondTensor > >
// {
//   using type = typename layout_result< ::std::remove_reference_t< BTE< decltype( ::std::declval< FirstTensor >().operator auto() ), decltype( ::std::declval< SecondTensor >().operator auto() ) > > >::type;
// };
// #else

// template < template < class, class > class BTE,
//            class                           FirstTensor,
//            class                           SecondTensor >
// struct layout_result< BTE< FirstTensor, SecondTensor > >
// {
// private:
//   template < class T, class U >
//   struct invalid_helper;
//   template < class T, class U >
//   struct readable_readable_helper
//   {
//     using type = ::std::conditional_t< ::std::is_same_v< typename T::layout_type, typename U::layout_type > &&
//                                          ! ::std::is_same_v< typename T::layout_type, ::std::layout_stride >,
//                                        typename T::layout_type,
//                                        default_layout >;
//   };
//   template < class T, class U >
//   struct readable_unevaluated_helper
//   {
//     using type = typename layout_result< BTE< FirstTensor, decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
//   };
//   template < class T, class U >
//   struct unevaluated_unevaluated_helper
//   {
//     using type = typename layout_result< BTE< decltype( ::std::declval< FirstTensor >().operator auto() ), decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
//   };
// public:
//   using type = ::std::conditional_t< ( LINALG_CONCEPTS::binary_tensor_expression_v< BTE< FirstTensor, SecondTensor > > &&
//                                        ( ::std::is_same_v< typename FirstTensor::layout_type, ::std::layout_right > ||
//                                          ::std::is_same_v< typename FirstTensor::layout_type, ::std::layout_left > ||
//                                          ::std::is_same_v< typename FirstTensor::layout_type, ::std::layout_stride > ) &&
//                                        ( ::std::is_same_v< typename SecondTensor::layout_type, ::std::layout_right > ||
//                                          ::std::is_same_v< typename SecondTensor::layout_type, ::std::layout_left > ||
//                                          ::std::is_same_v< typename SecondTensor::layout_type, ::std::layout_stride > ) ),
//                                      ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< FirstTensor >,
//                                                            ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< SecondTensor >,
//                                                                                  typename readable_readable_helper< FirstTensor, SecondTensor >::type,
//                                                                                  ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< SecondTensor >,
//                                                                                                        typename readable_unevaluated_helper< FirstTensor, SecondTensor >::type,
//                                                                                                        typename invalid_helper< FirstTensor, SecondTensor >::type > >,
//                                                            ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< FirstTensor >,
//                                                                                  ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< SecondTensor >,
//                                                                                                        typename readable_unevaluated_helper< SecondTensor, FirstTensor >::type,
//                                                                                                        ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< SecondTensor >,
//                                                                                                                              typename unevaluated_unevaluated_helper< FirstTensor, SecondTensor >::type,
//                                                                                                                              typename invalid_helper< FirstTensor, SecondTensor >::type > >,
//                                                                                  typename invalid_helper< FirstTensor, SecondTensor >::type > >,
//                                      typename invalid_helper< FirstTensor, SecondTensor >::type >;
// };
// #endif

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > > >
#endif
using layout_result_t = typename layout_result< Tensor >::type;

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_LAYOUT_RESULT_HPP
