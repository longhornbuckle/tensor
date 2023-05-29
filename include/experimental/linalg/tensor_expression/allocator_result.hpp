//==================================================================================================
//  File:       ACCESSOR_RESULT.hpp
//
//  Summary:    This header defines tensor expression traits:
//              allocator_result< Tensor >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_ALLOCATOR_RESULT_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_ALLOCATOR_RESULT_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

// Allocator result defines the resultant allocator of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
struct allocator_result
{
  using type = ::std::allocator< typename ::std::remove_reference_t< Tensor >::value_type >;
  [[nodiscard]] static inline constexpr type get_allocator( const Tensor&& ) noexcept { return type(); }
};

template < class Tensor >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > > &&
             LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< Tensor > > )
struct allocator_result< Tensor >
{
  using type = typename ::std::remove_reference_t< Tensor >::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( const Tensor&& t ) noexcept
  {
    if constexpr ( ! ::std::is_rvalue_reference_v<Tensor> )
    {
      return ::std::allocator_traits< typename ::std::remove_reference_t< Tensor >::allocator_type >::select_on_container_copy_construction( t.get_allocator() );
    }
    else
    {
      return t.get_allocator();
    }
  }
};
#else
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
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Tensor > >,
                                              dynamic_helper< Tensor >,
                                              ::std::conditional_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                    default_helper< Tensor >,
                                                                    invalid_helper< Tensor > > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const Tensor&& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Tensor > > )
    {
      return dynamic_helper< Tensor >::get_allocator( ::std::forward< const Tensor >( t ) );
    }
    else if constexpr ( LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > )
    {
      return default_helper< Tensor >::get_allocator( ::std::forward< const Tensor >( t ) );
    }
    else
    {
      return invalid_helper< Tensor >::get_allocator( ::std::forward< const Tensor >( t ) );
    }
  }
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
struct allocator_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = typename allocator_result< decltype( ::std::declval< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >().underlying() ) >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::negate_tensor_expression< Tensor >& t ) noexcept
    { return allocator_result< decltype( ::std::declval< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >().underlying() ) >::get_allocator( t.underlying() ); }
};

template < class Tensor,
           class Transpose >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
struct allocator_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = typename allocator_result< decltype( ::std::declval< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >().underlying() ) >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose >& t ) noexcept
    { return allocator_result< decltype( ::std::declval< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >().underlying() ) >::get_allocator( t.underlying() ); }
};

template < class Tensor,
           class Transpose >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
struct allocator_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose > >
{
  using type = typename allocator_result< decltype( ::std::declval< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor > >().underlying() ) >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor >& t ) noexcept
    { return allocator_result< decltype( ::std::declval< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor > >().underlying() ) >::get_allocator( t.underlying() ); }
};
#else
template < class Tensor,
           class Enable >
struct allocator_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable > >
{
  using type = typename allocator_result< decltype( ::std::declval< const LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable >& >().underlying() ) >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable >& t ) noexcept
  {
    return allocator_result< decltype( ::std::declval< const LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable >& >().underlying() ) >::get_allocator( t.underlying() );
  }
};

template < class Tensor,
           class Transpose,
           class Enable >
struct allocator_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable > >
{
  using type = typename allocator_result< decltype( ::std::declval< const LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable >& >().underlying() ) >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable >& t ) noexcept
  {
    return allocator_result< decltype( ::std::declval< const LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable >& >().underlying() ) >::get_allocator( t.underlying() );
  }
};

template < class Tensor,
           class Transpose,
           class Enable >
struct allocator_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose, Enable > >
{
  using type = typename allocator_result< decltype( ::std::declval< const LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose, Enable >& >().underlying() ) >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose, Enable >& t ) noexcept
  {
    return allocator_result< decltype( ::std::declval< const LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor, Transpose, Enable >& >().underlying() ) >::get_allocator( t.underlying() );
  }
};
#endif

// #ifdef LINALG_ENABLE_CONCEPTS
// template < template < class, class > class BTE,
//            class                           FirstTensor,
//            class                           SecondTensor >
//   requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
//              LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondTensor > > )
// struct allocator_result< BTE< FirstTensor, SecondTensor > >
// {
//   using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor< ::std::decay_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().first() ) > > ||
//                                                                   ! LINALG_CONCEPTS::dynamic_tensor< ::std::decay_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().second() ) > >,
//                                                                 decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().first() ),
//                                                                 decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().second() ) > >::type;
//   [[nodiscard]] static inline constexpr type get_allocator( const BTE< FirstTensor, SecondTensor >& t ) noexcept
//   {
//     if constexpr ( LINALG_CONCEPTS::dynamic_tensor< ::std::decay_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().first() ) > > ||
//                    ! LINALG_CONCEPTS::dynamic_tensor< ::std::decay_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().second() ) > > )
//     {
//       return allocator_result< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().first() ) >::get_allocator( t.first() );
//     }
//     else
//     {
//       return allocator_result< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().second() ) >::get_allocator( t.second() );
//     }
//   }
// };
// #else
// template < template < class, class, class > class BTE,
//            class                                  FirstTensor,
//            class                                  SecondTensor,
//            class                                  Enable >
// struct allocator_result< BTE< FirstTensor, SecondTensor, Enable > >
// {
// private:
//   template < class T >
//   struct invalid_helper;
//   template < class T >
//   struct valid_helper
//   {
//     using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().first() ) > > ||
//                                                                     ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().second() ) > >,
//                                                                   decltype( ::std::declval< T >().first() ),
//                                                                   decltype( ::std::declval< T >().second() ) > >::type;
//     [[nodiscard]] static inline constexpr type get_allocator( T&& t ) noexcept
//     {
//       if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().first() ) > > ||
//                      ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().second() ) > > )
//       {
//         return allocator_result< decltype( ::std::declval< T >().first() ) >::get_allocator( t.first() );
//       }
//       else
//       {
//         return allocator_result< decltype( ::std::declval< T >().second() ) >::get_allocator( t.second() );
//       }
//     }
//   };
// public:
//   using type = ::std::conditional_t< LINALG_CONCEPTS::binary_tensor_expression_v< BTE< FirstTensor, SecondTensor, Enable > >,
//                                      typename valid_helper< BTE< FirstTensor, SecondTensor, Enable > >::type,
//                                      typename invalid_helper< BTE< FirstTensor, SecondTensor, Enable > >::type >;
//   [[nodiscard]] static inline constexpr type get_allocator( const BTE< FirstTensor, SecondTensor, Enable >& t ) noexcept
//   {
//     if constexpr ( LINALG_CONCEPTS::binary_tensor_expression_v< BTE< FirstTensor, SecondTensor, Enable > > )
//     {
//       return valid_helper< BTE< FirstTensor, SecondTensor, Enable > >::get_allocator( t );
//     }
//     else
//     {
//       return invalid_helper< BTE< FirstTensor, SecondTensor, Enable > >::get_allocator( t );
//     }
//   }
// };
// #endif

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > > >
#endif
using allocator_result_t = typename allocator_result< Tensor >::type;

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_ALLOCATOR_RESULT_HPP
