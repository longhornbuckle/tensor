//==================================================================================================
//  File:       scalar_modulo.hpp
//
//  Summary:    This header defines:
//              LINALG::accessor_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
//              LINALG::allocator_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
//              LINALG::layout_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
//              LINALG::is_alias_assignable< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
//              LINALG_EXPRESSIONS_DETAIL::scalar_modulo_tensor_expression_traits< Tensor, Scalar >
//              LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar >
//              LINALG::operator % ( const Tensor& t, const Scalar& s )
//              LINALG::operator %= ( Tensor& t, const Scalar& s )
//==================================================================================================

#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_SCALAR_MODULO_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_SCALAR_MODULO_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//-------------------
//  Accessor Result
//-------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Scalar, class Tensor >
struct accessor_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< Tensor >::accessor_type,
                                  decltype( ::std::declval< Scalar >() % ::std::declval< typename ::std::remove_reference_t< Tensor >::value_type >() ) >;
};

#else

template < class Scalar, class Tensor, class Enable >
struct accessor_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar, Enable > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< Tensor >::accessor_type,
                                  decltype( ::std::declval< Scalar >() + ::std::declval< typename ::std::remove_reference_t< Tensor >::value_type >() ) >;
};

#endif

//--------------------
//  Allocator Result
//--------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Tensor, class Scalar >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
struct allocator_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
{
  using type = typename allocator_result< Tensor >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar >& t ) noexcept
  {
    return allocator_result< Tensor >::get_allocator( t.first() );
  }
};

#else

template < class Tensor, class Scalar, class Enable >
struct allocator_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar, Enable > >
{
  using type = typename allocator_result< Tensor >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar, Enable >& t ) noexcept
  {
    return allocator_result< Tensor >::get_allocator( t.first() );
  }
};

#endif

//-----------------
//  Layout Result
//-----------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Scalar, class Tensor >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
{
  using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride >,
                                     LINALG::default_layout,
                                     typename ::std::remove_reference_t< Tensor >::layout_type >;
};

template < class Scalar, class Tensor >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct layout_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< decltype( ::std::declval< Tensor >().operator auto() ), Scalar > > >::type;
};

#else

template < class Scalar, class Tensor, class Enable >
struct layout_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< T >::layout_type, ::std::layout_stride >,
                                       LINALG::default_layout,
                                       typename ::std::remove_reference_t< T >::layout_type >;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Scalar, decltype( ::std::declval< Tensor >().operator auto() ) > >::type;
  };
public:
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

//-----------------------
//  Is Alias Assignable
//-----------------------
template < class Tensor, class Scalar >
struct is_alias_assignable< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< Tensor, Scalar > > :
  public ::std::conditional_t< 
#ifdef LINALG_ENABLE_CONCEPTS
                               LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >,
#else
                               LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
#endif
                               is_alias_assignable< Tensor >,
                               ::std::true_type >
{ };

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

// Scalar Post-product tensor expression
template < class Tensor, class Scalar >
class scalar_modulo_tensor_expression_traits
{
  public:
    // Aliases
    using value_type   = decltype( ::std::declval< Scalar >() % ::std::declval< typename ::std::remove_reference_t< Tensor >::value_type >() );
    using index_type   = typename ::std::remove_reference_t< Tensor >::index_type;
    using size_type    = typename ::std::remove_reference_t< Tensor >::size_type;
    using extents_type = typename ::std::remove_reference_t< Tensor >::extents_type;
    using rank_type    = typename ::std::remove_reference_t< Tensor >::rank_type;
};

LINALG_EXPRESSIONS_DETAIL_END // expression detail namespace

LINALG_EXPRESSIONS_BEGIN // expressions namespace

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Scalar >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > > &&
             requires ( typename ::std::remove_reference_t< Tensor >::value_type v, Scalar s ) { v % s; } )
class scalar_modulo_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< scalar_modulo_tensor_expression< Tensor, Scalar >,
                                                                   LINALG_EXPRESSIONS_DETAIL::scalar_modulo_tensor_expression_traits< Tensor, Scalar > >
#else
template < class Tensor, class Scalar, typename Enable >
class scalar_modulo_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< scalar_modulo_tensor_expression< Tensor, Scalar, Enable >,
                                                                   LINALG_EXPRESSIONS_DETAIL::scalar_modulo_tensor_expression_traits< Tensor, Scalar > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = scalar_modulo_tensor_expression< Tensor, Scalar >;
    #else
    using self_type   = scalar_modulo_tensor_expression< Tensor, Scalar, Enable >;
    #endif
    using traits_type = LINALG_EXPRESSIONS_DETAIL::scalar_modulo_tensor_expression_traits< Tensor, Scalar >;
    using base_type   = LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr scalar_modulo_tensor_expression( Tensor&& t, Scalar&& s ) noexcept : t_(t), s_(s) {}
    constexpr scalar_modulo_tensor_expression& operator = ( const scalar_modulo_tensor_expression& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
    constexpr scalar_modulo_tensor_expression& operator = ( scalar_modulo_tensor_expression&& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
    // Aliases
    using value_type     = typename traits_type::value_type;
    using index_type     = typename traits_type::index_type;
    using size_type      = typename traits_type::size_type;
    using extents_type   = typename traits_type::extents_type;
    using rank_type      = typename traits_type::rank_type;
  private:
    template < class T, bool >
    struct helper
    {
      using type = LINALG::fs_tensor< typename T::value_type,
                                      typename T::extents_type,
                                      LINALG::layout_result_t< T >,
                                      LINALG::accessor_result_t< T > >;
    };
    template < class T >
    struct helper< T, false >
    {
      using type = LINALG::dr_tensor< typename T::value_type,
                                      typename T::extents_type,
                                      LINALG::layout_result_t< T >,
                                      typename T::extents_type,
                                      typename ::std::allocator_traits< LINALG::allocator_result_t< T > >::template rebind_alloc< typename T::value_type >,
                                      LINALG::accessor_result_t< T > >;
    };
  public:
    using evaluated_type = typename helper< self_type, ( extents_type::rank_dynamic() == 0 ) >::type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< Tensor >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return this->t_.extents(); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { return this->t_.extent(n); }
    // Binary tensor expression function
    [[nodiscard]] constexpr const Tensor& first() const noexcept { return this->t_; }
    [[nodiscard]] constexpr const Scalar& second() const noexcept { return this->s_; }
    // Scalar Pre-product
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( this->t_, indices ... ) % this->s_ ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
      { return LINALG_DETAIL::access( this->t_, indices ... ) % this->s_; }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( this->t_, indices ... ) % this->s_ ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
      { return LINALG_DETAIL::access( this->t_, indices ... ) % this->s_; }
    #endif
  private:
    // Define noexcept specification of conversion operator
    [[nodiscard]] static inline constexpr bool conversion_is_noexcept() noexcept
    {
      if constexpr( extents_type::rank_dynamic() == 0 )
      {
        return ::std::is_nothrow_constructible_v< evaluated_type,
                                                  const base_type& >;
      }
      else
      {
        return ::std::is_nothrow_constructible_v< evaluated_type,
                                                  const base_type&,
                                                  decltype( LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
      }
    }
  public:
    // Implicit conversion
    [[nodiscard]] constexpr operator evaluated_type() const noexcept( conversion_is_noexcept() )
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return evaluated_type( *static_cast< const base_type* >( this ) );
      }
      else
      {
        return evaluated_type( *static_cast< const base_type* >( this ), LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( *this ) ) );
      }
    }
    // Evaluated expression
    [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION auto evaluate() const noexcept( conversion_is_noexcept() )
    {
      return evaluated_type( *this );
    }
  private:
    // Data
    Tensor& t_;
    Scalar& s_;
};

LINALG_EXPRESSIONS_END // end expressions namespace

LINALG_BEGIN // linalg namespace

//--------------------------------------
//  Binary scalar modulo operator
//--------------------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< const T&, const S& >, const T&, const S& > >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator % ( const T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< const T&, const S& >, const T&, const S& >
#endif
{
  return LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< const T&, const S& >( t, s );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& >, T&, const S& > &&
                                            ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& > > ) > >
#endif
inline constexpr T&
operator %= ( T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& >, T&, const S& > &&
             ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& > > )
#endif
{
  return t = LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& >( t, s );
}

LINALG_END // linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_SCALAR_MODULO_HPP
