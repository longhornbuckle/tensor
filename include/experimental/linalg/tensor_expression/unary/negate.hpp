//==================================================================================================
//  File:       negate.hpp
//
//  Summary:    This header defines:
//              LINALG::accessor_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
//              LINALG::allocator_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
//              LINALG::layout_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
//              LINALG::is_alias_assignable< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
//              LINALG_EXPRESSIONS_DETAIL::negate_tensor_expression_traits< Tensor >
//              LINALG_EXPRESSIONS::negate_tensor_expression< Tensor >
//              LINALG::operator - ( const T& t )
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_NEGATE_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_NEGATE_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//-------------------
//  Accessor Result
//-------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Tensor >
  requires LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< Tensor >::accessor_type, decltype( - ::std::declval< typename ::std::remove_reference_t< Tensor >::value_type >() ) >;
};

template < class Tensor >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
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
    using type = typename ::std::conditional_t< LINALG_DETAIL::is_default_accessor_v< typename ::std::remove_reference_t< T >::accessor_type >,
                                                rebind_accessor< typename ::std::remove_reference_t< T >::accessor_type, decltype( - ::std::declval< typename ::std::remove_reference_t< T >::value_type >() ) >,
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

//--------------------
//  Allocator Result
//--------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
struct allocator_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = typename allocator_result< Tensor >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::negate_tensor_expression< Tensor >& t ) noexcept
    { return allocator_result< Tensor >::get_allocator( t.underlying() ); }
};

#else

template < class Tensor,
           class Enable >
struct allocator_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable > >
{
  using type = typename allocator_result< Tensor >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::negate_tensor_expression< Tensor, Enable >& t ) noexcept
  {
    return allocator_result< Tensor >::get_allocator( t.underlying() );
  }
};

#endif

//-----------------
//  Layout Result
//-----------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Tensor >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > >
{
  using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride >,
                                     LINALG::default_layout,
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
                                       LINALG::default_layout,
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

//-----------------------
//  Is Alias Assignable
//-----------------------
template < class Tensor >
struct is_alias_assignable< LINALG_EXPRESSIONS::negate_tensor_expression< Tensor > > :
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

/// @brief Traits class for negate tensor expression
/// @tparam Tensor underlying tensor type
template < class Tensor >
class negate_tensor_expression_traits
{
  public:
    using value_type   = ::std::remove_cv_t< decltype( - ::std::declval< typename ::std::remove_reference_t< Tensor >::value_type >() ) >;
    using index_type   = typename ::std::remove_reference_t< Tensor >::index_type;
    using size_type    = typename ::std::remove_reference_t< Tensor >::size_type;
    using extents_type = typename ::std::remove_reference_t< Tensor >::extents_type;
    using rank_type    = typename ::std::remove_reference_t< Tensor >::rank_type;
};

LINALG_EXPRESSIONS_DETAIL_END // expressions detail namespace

LINALG_EXPRESSIONS_BEGIN // expressions namespace

/// @brief Negate tensor expression defines the negation operation on a tensor
/// @tparam Tensor underlying tensor type
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
class negate_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< negate_tensor_expression< Tensor >,
                                                                  LINALG_EXPRESSIONS_DETAIL::negate_tensor_expression_traits< Tensor > >
#else
template < class Tensor, typename Enable >
class negate_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< negate_tensor_expression< Tensor, Enable >,
                                                                  LINALG_EXPRESSIONS_DETAIL::negate_tensor_expression_traits< Tensor > >
#endif
{
  private:
    // Aliases
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = negate_tensor_expression< Tensor >;
    #else
    using self_type   = negate_tensor_expression< Tensor, Enable >;
    #endif
    using traits_type = LINALG_EXPRESSIONS_DETAIL::negate_tensor_expression_traits< Tensor >;
    using base_type   = LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr negate_tensor_expression( Tensor&& t ) noexcept : t_(t) { }
    constexpr negate_tensor_expression& operator = ( const negate_tensor_expression& t ) noexcept { this->t_ = t.t_; }
    constexpr negate_tensor_expression& operator = ( negate_tensor_expression&& t ) noexcept { this->t_ = t.t_; }
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
    [[nodiscard]] constexpr extents_type extents() const noexcept { return t_.extents(); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { return t_.extent(n); }
    // Unary tensor expression function
    [[nodiscard]] constexpr decltype(auto) underlying() const noexcept { return this->t_; }
    // Negate
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( - LINALG_DETAIL::access( ::std::declval<Tensor&&>(), indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( this->t_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( - LINALG_DETAIL::access( ::std::declval<Tensor&&>(), indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( this->t_, indices ... ); }
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
                                                  decltype( LINALG::allocator_result< const self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
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
    Tensor&& t_;
};

LINALG_EXPRESSIONS_END // expressions namespace

LINALG_BEGIN // linalg namesapce

//---------------------------
//  Unary negation operator
//---------------------------
#ifdef LINALG_ENABLE_CONCEPTS
template < class T >
#else
template < class T, typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::negate_tensor_expression< const T& >, const T& > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator - ( const T& t ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::negate_tensor_expression< const T& >, const T& >
#endif
{
  return LINALG_EXPRESSIONS::negate_tensor_expression< const T& >( t );
}

LINALG_END // linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_NEGATE_HPP
