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

// Layout result defines the resultant layout of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct layout_result;

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::readable_tensor Tensor >
struct layout_result< negate_tensor_expression< Tensor > >
{
  using type = typename ::std::remove_reference_t< Tensor >::layout_type;
};

template < LINALG_CONCEPTS::unevaluated_tensor_expression Tensor >
struct layout_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};
#else
template < class Tensor >
struct layout_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
private :
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = typename ::std::remove_reference_t< T >::layout_type;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< ::std::remove_reference< T > >().operator auto() )::layout_type;
  };
public :
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                              readable_helper< Tensor >,
                                              ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                    unevaluated_helper< Tensor >,
                                                                    invalid_helper< Tensor > > >::type;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::readable_tensor Tensor >
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_stride > )
{
  using type = default_layout;
};

template < LINALG_CONCEPTS::unevaluated_tensor_expression Tensor >
struct layout_result< transpose_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};
#else
template < class Tensor >
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor > >
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
  using type = ::std::conditional_t< ( ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_right > ||
                                       ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_left > ||
                                       ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_stride > ),
                                     ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< Tensor >,
                                                           typename readable_helper< Tensor >::type,
                                                           ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< Tensor >,
                                                                                 typename unevaluated_helper< Tensor >::type,
                                                                                 typename invalid_helper< Tensor >::type > >,
                                     typename invalid_helper< Tensor >::type >;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::readable_tensor Tensor >
struct layout_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_stride > )
{
  using type = default_layout;
};

template < LINALG_CONCEPTS::unevaluated_tensor_expression Tensor >
struct layout_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};
#else
template < class Tensor >
struct layout_result< LINALG_EXPRESSIONS::conjugate_tensor_expression< Tensor > >
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
  using type = ::std::conditional_t< ( ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_right > ||
                                       ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_left > ||
                                       ::std::is_same_v< typename Tensor::layout_type, ::std::experimental::layout_stride > ),
                                     ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< Tensor >,
                                                           typename readable_helper< Tensor >::type,
                                                           ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< Tensor >,
                                                                                 typename unevaluated_helper< Tensor >::type,
                                                                                 typename invalid_helper< Tensor >::type > >,
                                     typename invalid_helper< Tensor >::type >;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class, class > LINALG_CONCEPTS::binary_tensor_expression BTE,
           LINALG_CONCEPTS::readable_tensor                                    FirstTensor,
           LINALG_CONCEPTS::readable_tensor                                    SecondTensor >
struct layout_result< BTE< FirstTensor, SecondTensor > >
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

template < template < class, class > LINALG_CONCEPTS::binary_tensor_expression BTE,
           LINALG_CONCEPTS::unevaluated_tensor_expression                      FirstTensor,
           LINALG_CONCEPTS::readable_tensor                                    SecondTensor >
struct layout_result< BTE< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< BTE< decltype( ::std::declval< FirstTensor >().operator auto() ), SecondTensor > >::type;
};

template < template < class, class > LINALG_CONCEPTS::binary_tensor_expression BTE,
           LINALG_CONCEPTS::readable_tensor                                    FirstTensor,
           LINALG_CONCEPTS::unevaluated_tensor_expression                      SecondTensor >
struct layout_result< BTE< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< BTE< FirstTensor, decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
};

template < template < class, class > LINALG_CONCEPTS::binary_tensor_expression BTE,
           LINALG_CONCEPTS::unevaluated_tensor_expression                      FirstTensor,
           LINALG_CONCEPTS::unevaluated_tensor_expression                      SecondTensor >
struct layout_result< BTE< FirstTensor, SecondTensor > >
{
  using type = typename layout_result< BTE< decltype( ::std::declval< FirstTensor >().operator auto() ), decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
};
#else

template < template < class, class > class BTE,
           class                           FirstTensor,
           class                           SecondTensor >
struct layout_result< BTE< FirstTensor, SecondTensor > >
{
private:
  template < class T, class U >
  struct invalid_helper;
  template < class T, class U >
  struct readable_readable_helper
  {
    using type = ::std::conditional_t< ::std::is_same_v< typename T::layout_type, typename U::layout_type > &&
                                         ! ::std::is_same_v< typename T::layout_type, ::std::experimental::layout_stride >,
                                       typename T::layout_type,
                                       default_layout >;
  };
  template < class T, class U >
  struct readable_unevaluated_helper
  {
    using type = typename layout_result< BTE< FirstTensor, decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
  };
  template < class T, class U >
  struct unevaluated_unevaluated_helper
  {
    using type = typename layout_result< BTE< decltype( ::std::declval< FirstTensor >().operator auto() ), decltype( ::std::declval< SecondTensor >().operator auto() ) > >::type;
  };
public:
  using type = ::std::conditional_t< ( LINALG_CONCEPTS::binary_tensor_expression_v< BTE< FirstTensor, SecondTensor > > &&
                                       ( ::std::is_same_v< typename FirstTensor::layout_type, ::std::experimental::layout_right > ||
                                         ::std::is_same_v< typename FirstTensor::layout_type, ::std::experimental::layout_left > ||
                                         ::std::is_same_v< typename FirstTensor::layout_type, ::std::experimental::layout_stride > ) &&
                                       ( ::std::is_same_v< typename SecondTensor::layout_type, ::std::experimental::layout_right > ||
                                         ::std::is_same_v< typename SecondTensor::layout_type, ::std::experimental::layout_left > ||
                                         ::std::is_same_v< typename SecondTensor::layout_type, ::std::experimental::layout_stride > ) ),
                                     ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< FirstTensor >,
                                                           ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< SecondTensor >,
                                                                                 typename readable_readable_helper< FirstTensor, SecondTensor >::type,
                                                                                 ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< SecondTensor >,
                                                                                                       typename readable_unevaluated_helper< FirstTensor, SecondTensor >::type,
                                                                                                       typename invalid_helper< FirstTensor, SecondTensor >::type > >,
                                                           ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< FirstTensor >,
                                                                                 ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< SecondTensor >,
                                                                                                       typename readable_unevaluated_helper< SecondTensor, FirstTensor >::type,
                                                                                                       ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< SecondTensor >,
                                                                                                                             typename unevaluated_unevaluated_helper< FirstTensor, SecondTensor >::type,
                                                                                                                             typename invalid_helper< FirstTensor, SecondTensor >::type > >,
                                                                                 typename invalid_helper< FirstTensor, SecondTensor >::type > >,
                                     typename invalid_helper< FirstTensor, SecondTensor >::type >;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > > >
#endif
using layout_result_t = typename layout_result< Tensor >::type;


// Accessor result defines the resultant accessor of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct accessor_result;

template < class T >
class is_default_accessor : public ::std::false_type { };

template < class T >
class is_default_accessor< ::std::experimental::default_accessor< T > > : public ::std::true_type { };

template < class T >
inline constexpr bool is_default_accessor_v = is_default_accessor< T >::value;

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class > LINALG_CONCEPTS::unary_tensor_expression UTE,
           LINALG_CONCEPTS::readable_tensor                            Tensor >
struct accessor_result< UTA< Tensor > >
{
  using type = typename Tensor::accessor_type;
};

template < template < class > LINALG_CONCEPTS::unary_tensor_expression UTE,
           LINALG_CONCEPTS::unevaluated_tensor_expression              Tensor >
struct accessor_result< UTA< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};
#else
template < template < class, class > class UTE,
           class                           Tensor,
           class                           Enable >
struct accessor_result< UTE< Tensor, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = typename ::std::remove_reference_t< T >::accessor_type;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< T >().operator auto() )::accessor_type;
  };
public:
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::unary_tensor_expression_v< ::std::decay_t< UTE< Tensor, Enable > > >,
                                              ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                                                    readable_helper< Tensor >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                                          unevaluated_helper< Tensor >,
                                                                                          invalid_helper< Tensor > > >,
                                              invalid_helper< UTE< Tensor, Enable > > >::type;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class, class > LINALG_CONCEPTS::binary_tensor_expression BTE,
           LINALG_CONCEPTS::tensor_expression                                  FirstTensor,
           LINALG_CONCEPTS::tensor_expression                                  SecondTensor >
struct accessor_result< BTE< FirstTensor, SecondTensor > >
  requires ( is_default_accessor_v< typename FirstTensor::accessor_type > &&
             is_default_accessor_v< typename SecondTensor::accessor_type > )
{
  using type = ::std::experimental::default_accessor< decltype( ::std::declval< typename FirstTensor::value_type >() + ::std::declval< typename SecondTensor::value_type >() ) >;
};
#else
template < template < class, class, class Enable > class BTE,
           class                                         FirstTensor,
           class                                         SecondTensor,
           class                                         Enable >
struct accessor_result< BTE< FirstTensor, SecondTensor, Enable > >
{
private:
  template < class T, class U >
  struct invalid_helper;
  template < class T, class U >
  struct default_helper
  {
    using type = ::std::conditional_t< ( is_default_accessor_v< typename ::std::remove_reference_t< T >::accessor_type > &&
                                         is_default_accessor_v< typename ::std::remove_reference_t< U >::accessor_type > ),
                                       ::std::experimental::default_accessor< decltype( ::std::declval< typename ::std::remove_reference_t< T >::value_type >() + ::std::declval< typename ::std::remove_reference_t< U >::value_type >() ) >,
                                       typename invalid_helper< T, U >::type >;
  };
public:
  using type = typename ::std::conditional_t< ( LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< FirstTensor > > &&
                                                LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< SecondTensor > > ),
                                              default_helper< FirstTensor, SecondTensor >,
                                              invalid_helper< FirstTensor, SecondTensor > >::type;
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
using accessor_result_t = typename accessor_result< Tensor >::type;


// Allocator result defines the resultant allocator of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
struct allocator_result
{
  using type = ::std::allocator< typename Tensor::value_type >;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& ) noexcept { return type(); }
};

template < LINALG_CONCEPTS::dynamic_tensor Tensor >
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
    [[nodiscard]] static inline constexpr type get_allocator( T&& ) noexcept { return type(); }
  };
  template < class T >
  struct dynamic_helper
  {
    using type = typename ::std::remove_reference_t< T >::allocator_type;
    [[nodiscard]] static inline constexpr type get_allocator( T&& t ) noexcept
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
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Tensor > > )
    {
      return dynamic_helper< Tensor >::get_allocator( t );
    }
    else if constexpr ( LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > )
    {
      return default_helper< Tensor >::get_allocator( t );
    }
    else
    {
      return invalid_helper< Tensor >::get_allocator( t );
    }
  }
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class > LINALG_CONCEPTS::unary_tensor_expression UTE,
           LINALG_CONCEPTS::tensor_expression                          Tensor >
struct allocator_result< UTE< Tensor > >
{
  using type = typename allocator_result< decltype( ::std::declval< UTE< Tensor > >.underlying() ) >::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( UTE< Tensor >&& t ) noexcept { return allocator_result< decltype( ::std::declval< UTE< Tensor > >.underlying() ) >::get_allocator( t.underlying(); ) }
};
#else
template < template < class, class > class UTE,
           class                           Tensor,
           class                           Enable >
struct allocator_result< UTE< Tensor, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct valid_helper
  {
    using type = typename allocator_result< decltype( ::std::declval< T >().underlying() ) >::type;
    [[nodiscard]] static inline constexpr type get_allocator( T&& t ) noexcept { return allocator_result< decltype( ::std::declval< T >().underlying() ) >::get_allocator( t.underlying() ); }
  };
public:
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::unary_tensor_expression_v< UTE< Tensor, Enable > >,
                                              valid_helper< UTE< Tensor, Enable > >,
                                              invalid_helper< UTE< Tensor, Enable > > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const UTE< Tensor, Enable >& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::unary_tensor_expression_v< UTE< Tensor, Enable > > )
    {
      return valid_helper< const UTE< Tensor, Enable > >::get_allocator( ::std::forward< const UTE< Tensor, Enable > >( t ) );
    }
    else
    {
      return invalid_helper< const UTE< Tensor, Enable > >::get_allocator( ::std::forward< const UTE< Tensor, Enable > >( t ) );
    }
  }
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class, class > LINALG_CONCEPTS::binary_tensor_expression BTE,
           LINALG_CONCEPTS::tensor_expression                                  FirstTensor,
           LINALG_CONCEPTS::tensor_expression                                  SecondTensor >
struct allocator_result< BTE< FirstTensor, SecondTensor > >
{
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor< ::std::remove_cv_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >.first() ) > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_cv_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >.second() ) > >,
                                                                decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().first() ),
                                                                decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().second() ) > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( BTE< FirstTensor, SecondTensor >&& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor< ::std::remove_cv_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().first() ) > > ||
                   ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_cv_t< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().second() ) > > )
    {
      return allocator_result< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().first() ) >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< decltype( ::std::declval< BTE< FirstTensor, SecondTensor > >().second() ) >::get_allocator( t.second() );
    }
  }
};
#else
template < template < class, class, class > class BTE,
           class                                  FirstTensor,
           class                                  SecondTensor,
           class                                  Enable >
struct allocator_result< BTE< FirstTensor, SecondTensor, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct valid_helper
  {
    using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().first() ) > > ||
                                                                    ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().second() ) > >,
                                                                  decltype( ::std::declval< T >().first() ),
                                                                  decltype( ::std::declval< T >().second() ) > >::type;
    [[nodiscard]] static inline constexpr type get_allocator( T&& t ) noexcept
    {
      if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().first() ) > > ||
                     ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::decay_t< decltype( ::std::declval< T >().second() ) > > )
      {
        return allocator_result< decltype( ::std::declval< T >().first() ) >::get_allocator( t.first() );
      }
      else
      {
        return allocator_result< decltype( ::std::declval< T >().second() ) >::get_allocator( t.second() );
      }
    }
  };
public:
  using type = ::std::conditional_t< LINALG_CONCEPTS::binary_tensor_expression_v< BTE< FirstTensor, SecondTensor, Enable > >,
                                     typename valid_helper< BTE< FirstTensor, SecondTensor, Enable > >::type,
                                     typename invalid_helper< BTE< FirstTensor, SecondTensor, Enable > >::type >;
  [[nodiscard]] static inline constexpr type get_allocator( BTE< FirstTensor, SecondTensor, Enable >&& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::binary_tensor_expression_v< BTE< FirstTensor, SecondTensor, Enable > > )
    {
      return valid_helper< BTE< FirstTensor, SecondTensor, Enable > >::get_allocator( t );
    }
    else
    {
      return invalid_helper< BTE< FirstTensor, SecondTensor, Enable > >::get_allocator( t );
    }
  }
};
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > > >
#endif
using allocator_result_t = typename allocator_result< Tensor >::type;

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_TENSOR_EXPRESSION_TRAITS_HPP
