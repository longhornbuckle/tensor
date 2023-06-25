//==================================================================================================
//  File:       vector_product.hpp
//
//  Summary:    This header defines:
//              LINALG::accessor_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
//              LINALG::allocator_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
//              LINALG::layout_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
//              LINALG_EXPRESSIONS_DETAIL::outer_product_expression_traits< FirstVector, SecondVector >
//              LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector >
//              LINALG::outer_product( const V1& v1, const V2& v2 )
//              LINALG::inner_product( const V1& v1, const V2& v2 )
//==================================================================================================

#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_VECTOR_PRODUCT_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_VECTOR_PRODUCT_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//-------------------
//  Accessor Result
//-------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstVector, class SecondVector >
struct accessor_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< FirstVector >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< FirstVector >::value_type >() * ::std::declval< typename ::std::remove_reference_t< SecondVector >::value_type >() ) >;
};

#else

template < class FirstVector, class SecondVector, class Enable >
struct accessor_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector, Enable > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< FirstVector >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< FirstVector >::value_type >() * ::std::declval< typename ::std::remove_reference_t< SecondVector >::value_type >() ) >;
};

#endif

//--------------------
//  Allocator Result
//--------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstVector, class SecondVector >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondVector > > )
struct allocator_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
{
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< FirstVector > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< SecondVector > >,
                                                                FirstVector,
                                                                SecondVector > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector >& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< FirstVector > > ||
                   ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< SecondVector > > )
    {
      return allocator_result< FirstVector >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< SecondVector >::get_allocator( t.second() );
    }
  }
};

#else

template < class FirstVector, class SecondVector, class Enable >
struct allocator_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector, Enable > >
{
private:
  using T = LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector, Enable >;
public:
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< FirstVector > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< SecondVector > >,
                                                                FirstVector,
                                                                SecondVector > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const T& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< FirstVector > > ||
                    ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< SecondVector > > )
    {
      return allocator_result< FirstVector >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< SecondVector >::get_allocator( t.second() );
    }
  }
};

#endif

//-----------------
//  Layout Result
//-----------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstVector, class SecondVector >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondVector > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, ::std::layout_stride > ) &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< SecondVector >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< SecondVector >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< SecondVector >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
{
  using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, typename ::std::remove_reference_t< SecondVector >::layout_type > &&
                                       ! ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, ::std::layout_stride >,
                                     typename ::std::remove_reference_t< FirstVector >::layout_type,
                                     LINALG::default_layout >;
};

template < class FirstVector, class SecondVector >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondVector > > )
struct layout_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::outer_product_expression< decltype( ::std::declval< FirstVector >().operator auto() ), SecondVector > > >::type;
};

template < class FirstVector, class SecondVector >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondVector > > )
struct layout_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, decltype( ::std::declval< SecondVector >().operator auto() ) > > >::type;
};

template < class FirstVector, class SecondVector >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondVector > > )
struct layout_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::outer_product_expression< decltype( ::std::declval< FirstVector >().operator auto() ), decltype( ::std::declval< SecondVector >().operator auto() ) > > >::type;
};

#else

template < class FirstVector, class SecondVector, class Enable >
struct layout_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, SecondVector, Enable > >
{
private:
  template < class T, class U >
  struct invalid_helper;
  template < class T, class U >
  struct readable_readable_helper
  {
    using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< T >::layout_type, typename ::std::remove_reference_t< U >::layout_type > &&
                                         ! ::std::is_same_v< typename ::std::remove_reference_t< T >::layout_type, ::std::layout_stride >,
                                       typename ::std::remove_reference_t< T >::layout_type,
                                       LINALG::default_layout >;
  };
  template < class T, class U >
  struct readable_unevaluated_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::outer_product_expression< FirstVector, decltype( ::std::declval< SecondVector >().evaluate() ) > >::type;
  };
  template < class T, class U >
  struct unevaluated_unevaluated_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::outer_product_expression< decltype( ::std::declval< FirstVector >().evaluate() ), decltype( ::std::declval< SecondVector >().operator auto() ) > >::type;
  };
public:
  using type = typename ::std::conditional_t< ( ( ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< FirstVector >::layout_type, ::std::layout_stride > ) &&
                                                ( ::std::is_same_v< typename ::std::remove_reference_t< SecondVector >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< SecondVector >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< SecondVector >::layout_type, ::std::layout_stride > ) ),
                                              typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< FirstVector > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< SecondVector > >,
                                                                                          readable_readable_helper< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondVector > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > >,
                                                                                                                invalid_helper< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > > > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< FirstVector > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< SecondVector > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< SecondVector >, ::std::remove_reference_t< FirstVector > >,
                                                                                                                ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondVector > >,
                                                                                                                                      unevaluated_unevaluated_helper< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > >,
                                                                                                                                      invalid_helper< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > > > >,
                                                                                          invalid_helper< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > > > >,
                                              invalid_helper< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > > >::type;
};

#endif

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

// Outer product expression traits
template < class FirstVector, class SecondVector >
class outer_product_expression_traits
{
  public:
    // Aliases
    using value_type   = decltype( ::std::declval< typename ::std::remove_reference_t< FirstVector >::value_type >() * ::std::declval< typename ::std::remove_reference_t< SecondVector >::value_type >() );
    using index_type   = ::std::common_type_t< typename ::std::remove_reference_t< FirstVector >::index_type, typename ::std::remove_reference_t< SecondVector >::index_type >;
    using size_type    = ::std::common_type_t< typename ::std::remove_reference_t< FirstVector >::size_type, typename ::std::remove_reference_t< SecondVector >::size_type >;
    using extents_type = ::std::extents< ::std::common_type_t< typename ::std::remove_reference_t< FirstVector >::extents_type::index_type,
                                                               typename ::std::remove_reference_t< SecondVector >::extents_type::index_type >,
                                         ::std::remove_reference_t< FirstVector >::extents_type::static_extent( 0 ),
                                         ::std::remove_reference_t< SecondVector >::extents_type::static_extent( 0 ) >;
    using rank_type    = ::std::common_type_t< typename ::std::remove_reference_t< FirstVector >::rank_type, typename ::std::remove_reference_t< SecondVector >::rank_type >;
};

LINALG_EXPRESSIONS_DETAIL_END // expression detail namespace

LINALG_EXPRESSIONS_BEGIN // expressions namespace

#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstVector, class SecondVector >
  requires ( LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< SecondVector > > &&
            ( requires ( const typename ::std::remove_reference_t< FirstVector >::value_type& v1, const typename ::std::remove_reference_t< SecondVector >::value_type& v2 ) { v1 * v2; } ) )
class outer_product_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< outer_product_expression< FirstVector, SecondVector >,
                                                                   LINALG_EXPRESSIONS_DETAIL::outer_product_expression_traits< FirstVector, SecondVector > >
#else
template < class FirstVector, class SecondVector, typename Enable >
class outer_product_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< outer_product_expression< FirstVector, SecondVector, Enable >,
                                                                   LINALG_EXPRESSIONS_DETAIL::outer_product_expression_traits< FirstVector, SecondVector > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = outer_product_expression< FirstVector, SecondVector >;
    #else
    using self_type   = outer_product_expression< FirstVector, SecondVector, Enable >;
    #endif
    using traits_type = LINALG_EXPRESSIONS_DETAIL::outer_product_expression_traits< FirstVector, SecondVector >;
    using base_type   = LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr outer_product_expression( FirstVector&& v1, SecondVector&& v2 ) noexcept : v1_(v1), v2_(v2) {}
    constexpr outer_product_expression& operator = ( const outer_product_expression& t ) noexcept { this->v1_ = t.v1_; this->v2_ = t.v2_; }
    constexpr outer_product_expression& operator = ( outer_product_expression&& t ) noexcept { this->v1_ = t.v1_; this->v2_ = t.v2_; }
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
    [[nodiscard]] static constexpr rank_type rank() noexcept { return rank_type( 2 ); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return extents_type( this->v1_.extent(0), this->v2_.extent(0) ); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { if ( n == 0 ) { return this->v1_.extent(0); } else { return this->v2_.extent(n-1); } }
    // Binary tensor expression function
    [[nodiscard]] constexpr const FirstVector& first() const noexcept { return this->v1_; }
    [[nodiscard]] constexpr const SecondVector& second() const noexcept { return this->v2_; }
  private:
    // Access
    [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION value_type access( index_type index1, index_type index2 ) const
      noexcept( noexcept( LINALG_DETAIL::access( ::std::declval< FirstVector >(), index1 ) ) &&
                noexcept( LINALG_DETAIL::access( ::std::declval< SecondVector >(), index2 ) ) )
    {
      return LINALG_DETAIL::access( this->v1_, index1 ) * LINALG_DETAIL::access( this->v2_, index2 );
    }
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
    // Outer product
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( this->access( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
      { return this->access( indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( this->access( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
      { return this->access( indices ... ); }
    #endif
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
    FirstVector&  v1_;
    SecondVector& v2_;
};

LINALG_EXPRESSIONS_END // end expressions namespace

LINALG_BEGIN // linalg namespace

//-----------------------------
//  Outer product operators
//-----------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class V1, class V2 >
#else
template < class V1, class V2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::outer_product_expression< const V1&, const V2& >, const V1&, const V2& > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
outer_prod( const V1& v1, const V2& v2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::outer_product_expression< const V1&, const V2& >, const V1&, const V2& >
#endif
{
  return LINALG_EXPRESSIONS::outer_product_expression< const V1&, const V2& >( v1, v2 );
}

//-----------------------------
//  Inner product operators
//-----------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstVector, class SecondVector >
#else
template < class FirstVector, class SecondVector,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< FirstVector > > &&
                                          LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< SecondVector > > &&
                                          LINALG_CONCEPTS::elements_are_multiplicative_v< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > > &&
                                          LINALG_CONCEPTS::may_have_equal_extents_v< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > > > >
#endif
[[nodiscard]] inline constexpr auto
inner_prod( const FirstVector& v1, const SecondVector& v2 ) noexcept( LINALG_DETAIL::extents_are_equal_v< typename ::std::remove_reference_t< FirstVector >::extents_type, typename ::std::remove_reference_t< SecondVector >::extents_type > )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< SecondVector > > &&
            ( requires ( const typename ::std::remove_reference_t< FirstVector >::value_type& v1, const typename ::std::remove_reference_t< SecondVector >::value_type& v2 ) { v1 * v2; } ) &&
            LINALG_DETAIL::extents_may_be_equal_v< typename ::std::remove_reference_t< FirstVector >::extents_type, typename ::std::remove_reference_t< SecondVector >::extents_type > )
#endif
{
  if constexpr ( ! LINALG_DETAIL::extents_are_equal_v< typename ::std::remove_reference_t< FirstVector >::extents_type, typename ::std::remove_reference_t< SecondVector >::extents_type > )
  {
    using common_index_type = ::std::common_type_t< typename ::std::remove_reference_t< FirstVector >::index_type,
                                                    typename ::std::remove_reference_t< SecondVector >::index_type >;
    if ( static_cast< common_index_type >( v1.extent( 0 ) ) != static_cast< common_index_type >( v2.extent( 0 ) ) ) LINALG_UNLIKELY
    {
      throw ::std::length_error( "Vector extents are incompatable." );
    }
  }
  decltype( LINALG_DETAIL::access( v1, 0 ) * LINALG_DETAIL::access( v2, 0 ) ) val { 0 };
  if constexpr ( ::std::remove_reference_t< SecondVector >::static_extent(0) == ::std::dynamic_extent )
  {
    for ( typename ::std::remove_reference_t< FirstVector >::index_type count = 0; count < v1.extent(0); ++count )
    {
      val += LINALG_DETAIL::access( v1, count ) * LINALG_DETAIL::access( v2, count );
    }
  }
  else
  {
    for ( typename ::std::remove_reference_t< SecondVector >::index_type count = 0; count < v2.extent(0); ++count )
    {
      val += LINALG_DETAIL::access( v1, count ) * LINALG_DETAIL::access( v2, count );
    }
  }
  return val;
}

LINALG_END // linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_VECTOR_PRODUCT_HPP
