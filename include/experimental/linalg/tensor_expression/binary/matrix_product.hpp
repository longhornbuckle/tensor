//==================================================================================================
//  File:       matrix_product.hpp
//
//  Summary:    This header defines:
//              LINALG::accessor_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
//              LINALG::allocator_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
//              LINALG::layout_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
//              LINALG_EXPRESSIONS_DETAIL::matrix_product_expression_traits< FirstMatrix, SecondMatrix >
//              LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix >
//              LINALG::operator * ( const M1& m1, const M2& m2 )
//              LINALG::operator *= ( M1& m1, const M2& m2 )
//==================================================================================================

#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_MATRIX_PRODUCT_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_MATRIX_PRODUCT_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//-------------------
//  Accessor Result
//-------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstMatrix, class SecondMatrix >
struct accessor_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< FirstMatrix >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< FirstMatrix >::value_type >() * ::std::declval< typename ::std::remove_reference_t< SecondMatrix >::value_type >() ) >;
};

#else

template < class FirstMatrix, class SecondMatrix, class Enable >
struct accessor_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix, Enable > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< FirstMatrix >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< FirstMatrix >::value_type >() * ::std::declval< typename ::std::remove_reference_t< SecondMatrix >::value_type >() ) >;
};

#endif

//--------------------
//  Allocator Result
//--------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstMatrix, class SecondMatrix >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstMatrix > > &&
             LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondMatrix > > )
struct allocator_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
{
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< FirstMatrix > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< SecondMatrix > >,
                                                                FirstMatrix,
                                                                SecondMatrix > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix >& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< FirstMatrix > > ||
                   ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< SecondMatrix > > )
    {
      return allocator_result< FirstMatrix >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< SecondMatrix >::get_allocator( t.second() );
    }
  }
};

#else

template < class FirstMatrix, class SecondMatrix, class Enable >
struct allocator_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix, Enable > >
{
private:
  using T = LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix, Enable >;
public:
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< FirstMatrix > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< SecondMatrix > >,
                                                                FirstMatrix,
                                                                SecondMatrix > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const T& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< FirstMatrix > > ||
                    ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< SecondMatrix > > )
    {
      return allocator_result< FirstMatrix >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< SecondMatrix >::get_allocator( t.second() );
    }
  }
};

#endif

//-----------------
//  Layout Result
//-----------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class FirstMatrix, class SecondMatrix >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstMatrix > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondMatrix > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, ::std::layout_stride > ) &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< SecondMatrix >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< SecondMatrix >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< SecondMatrix >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
{
  using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, typename ::std::remove_reference_t< SecondMatrix >::layout_type > &&
                                       ! ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, ::std::layout_stride >,
                                     typename ::std::remove_reference_t< FirstMatrix >::layout_type,
                                     LINALG::default_layout >;
};

template < class FirstMatrix, class SecondMatrix >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstMatrix > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< SecondMatrix > > )
struct layout_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::matrix_product_expression< decltype( ::std::declval< FirstMatrix >().operator auto() ), SecondMatrix > > >::type;
};

template < class FirstMatrix, class SecondMatrix >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< FirstMatrix > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondMatrix > > )
struct layout_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, decltype( ::std::declval< SecondMatrix >().operator auto() ) > > >::type;
};

template < class FirstMatrix, class SecondMatrix >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< FirstMatrix > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< SecondMatrix > > )
struct layout_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::matrix_product_expression< decltype( ::std::declval< FirstMatrix >().operator auto() ), decltype( ::std::declval< SecondMatrix >().operator auto() ) > > >::type;
};

#else

template < class FirstMatrix, class SecondMatrix, class Enable >
struct layout_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, SecondMatrix, Enable > >
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
    using type = typename layout_result< LINALG_EXPRESSIONS::matrix_product_expression< FirstMatrix, decltype( ::std::declval< SecondMatrix >().operator auto() ) > >::type;
  };
  template < class T, class U >
  struct unevaluated_unevaluated_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::matrix_product_expression< decltype( ::std::declval< FirstMatrix >().operator auto() ), decltype( ::std::declval< SecondMatrix >().operator auto() ) > >::type;
  };
public:
  using type = typename ::std::conditional_t< ( ( ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< FirstMatrix >::layout_type, ::std::layout_stride > ) &&
                                                ( ::std::is_same_v< typename ::std::remove_reference_t< SecondMatrix >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< SecondMatrix >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< SecondMatrix >::layout_type, ::std::layout_stride > ) ),
                                              typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< FirstMatrix > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< SecondMatrix > >,
                                                                                          readable_readable_helper< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondMatrix > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > >,
                                                                                                                invalid_helper< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > > > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< FirstMatrix > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< SecondMatrix > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< SecondMatrix >, ::std::remove_reference_t< FirstMatrix > >,
                                                                                                                ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< SecondMatrix > >,
                                                                                                                                      unevaluated_unevaluated_helper< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > >,
                                                                                                                                      invalid_helper< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > > > >,
                                                                                          invalid_helper< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > > > >,
                                              invalid_helper< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > > >::type;
};

#endif

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

// Matrix product expression traits
template < class FirstMatrix, class SecondMatrix >
class matrix_product_expression_traits
{
  public:
    // Aliases
    using value_type   = decltype( ::std::declval< typename ::std::remove_reference_t< FirstMatrix >::value_type >() * ::std::declval< typename ::std::remove_reference_t< SecondMatrix >::value_type >() );
    using index_type   = ::std::common_type_t< typename ::std::remove_reference_t< FirstMatrix >::index_type, typename ::std::remove_reference_t< SecondMatrix >::index_type >;
    using size_type    = ::std::common_type_t< typename ::std::remove_reference_t< FirstMatrix >::size_type, typename ::std::remove_reference_t< SecondMatrix >::size_type >;
    using extents_type = ::std::extents< ::std::common_type_t< typename ::std::remove_reference_t< FirstMatrix >::extents_type::index_type,
                                                               typename ::std::remove_reference_t< SecondMatrix >::extents_type::index_type >,
                                         ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent( 0 ),
                                         ::std::remove_reference_t< SecondMatrix >::extents_type::static_extent( 1 ) >;
    using rank_type    = ::std::common_type_t< typename ::std::remove_reference_t< FirstMatrix >::rank_type, typename ::std::remove_reference_t< SecondMatrix >::rank_type >;
};

LINALG_EXPRESSIONS_DETAIL_END // expression detail namespace

LINALG_EXPRESSIONS_BEGIN // expressions namespace

#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstMatrix, class SecondMatrix >
  requires ( LINALG_CONCEPTS::matrix_expression< ::std::remove_reference_t< FirstMatrix > > &&
             LINALG_CONCEPTS::matrix_expression< ::std::remove_reference_t< SecondMatrix > > &&
            ( requires ( const typename ::std::remove_reference_t< FirstMatrix >::value_type& v1, const typename ::std::remove_reference_t< SecondMatrix >::value_type& v2 ) { v1 * v2; } ) &&
             ( ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent(1) == ::std::remove_reference_t< SecondMatrix >::extents_type::static_extent(0) ) ||
               ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent(1) == ::std::dynamic_extent ) ||
               ( ::std::remove_reference_t< SecondMatrix >::extents_type::static_extent(0) == ::std::dynamic_extent ) ) )
class matrix_product_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< matrix_product_expression< FirstMatrix, SecondMatrix >,
                                                                   LINALG_EXPRESSIONS_DETAIL::matrix_product_expression_traits< FirstMatrix, SecondMatrix > >
#else
template < class FirstMatrix, class SecondMatrix, typename Enable >
class matrix_product_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< matrix_product_expression< FirstMatrix, SecondMatrix, Enable >,
                                                                   LINALG_EXPRESSIONS_DETAIL::matrix_product_expression_traits< FirstMatrix, SecondMatrix > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = matrix_product_expression< FirstMatrix, SecondMatrix >;
    #else
    using self_type   = matrix_product_expression< FirstMatrix, SecondMatrix, Enable >;
    #endif
    using traits_type = LINALG_EXPRESSIONS_DETAIL::matrix_product_expression_traits< FirstMatrix, SecondMatrix >;
    using base_type   = LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr matrix_product_expression( FirstMatrix&& m1, SecondMatrix&& m2 )
      noexcept( ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent( 1 ) == ::std::remove_reference_t< SecondMatrix >::extents_type::static_extent( 0 ) ) &&
                ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent( 1 ) != ::std::dynamic_extent ) ) :
      m1_(m1), m2_(m2)
    {
      if constexpr ( !( ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent( 1 ) == ::std::remove_reference_t< SecondMatrix >::extents_type::static_extent( 0 ) ) &&
                        ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent( 1 ) != ::std::dynamic_extent ) ) )
      {
        using common_index_type = ::std::common_type_t< typename ::std::remove_reference_t< FirstMatrix >::index_type,
                                                        typename ::std::remove_reference_t< SecondMatrix >::index_type >;
        if ( static_cast< common_index_type >( m1.extent( 1 ) ) != static_cast< common_index_type >( m2.extent( 0 ) ) ) LINALG_UNLIKELY
        {
          throw ::std::length_error( "Matrix extents are incompatable." );
        }
      }
    }
    constexpr matrix_product_expression& operator = ( const matrix_product_expression& t ) noexcept { this->m1_ = t.m1_; this->m2_ = t.m2_; }
    constexpr matrix_product_expression& operator = ( matrix_product_expression&& t ) noexcept { this->m1_ = t.m1_; this->m2_ = t.m2_; }
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
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< FirstMatrix >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return extents_type( this->m1_.extent(0), this->m2_.extent(1) ); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { if ( n == 0 ) { return this->m1_.extent(0); } else { return this->m2_.extent(n); } }
    // Binary tensor expression function
    [[nodiscard]] constexpr const FirstMatrix& first() const noexcept { return this->m1_; }
    [[nodiscard]] constexpr const SecondMatrix& second() const noexcept { return this->m2_; }
  private:
    // Access
    [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION value_type access( index_type index1, index_type index2 ) const
      noexcept( noexcept( LINALG_DETAIL::access( ::std::declval< FirstMatrix >(), index1, ::std::declval< index_type >() ) ) &&
                noexcept( LINALG_DETAIL::access( ::std::declval< SecondMatrix >(), ::std::declval< index_type >(), index2 ) ) )
    {
      value_type val { 0 };
      if constexpr ( ::std::remove_reference_t< FirstMatrix >::static_extent(1) != ::std::dynamic_extent )
      {
        for ( typename ::std::remove_reference_t< FirstMatrix >::index_type count = 0; count < this->m1_.extent(1); ++count )
        {
          val += LINALG_DETAIL::access( this->m1_, index1, count ) * LINALG_DETAIL::access( this->m2_, count, index2 );
        }
      }
      else
      {
        for ( typename ::std::remove_reference_t< SecondMatrix >::index_type count = 0; count < this->m2_.extent(0); ++count )
        {
          val += LINALG_DETAIL::access( this->m1_, index1, count ) * LINALG_DETAIL::access( this->m2_, count, index2 );
        }
      }
      return val;
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
    // Matrix product
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
    FirstMatrix&  m1_;
    SecondMatrix& m2_;
};

LINALG_EXPRESSIONS_END // end expressions namespace

LINALG_BEGIN // linalg namespace

//-----------------------------
//  Matrix product operators
//-----------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class M1, class M2 >
#else
template < class M1, class M2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< const M1&, const M2& >, const M1&, const M2& > >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const M1& m1, const M2& m2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< const M1&, const M2& >, const M1&, const M2& >
#endif
{
  return LINALG_EXPRESSIONS::matrix_product_expression< const M1&, const M2& >( m1, m2 );
}

//----------------------------------------
//  Matrix product assignment operators
//----------------------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class M1, class M2 >
#else
template < class M1, class M2,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& >, M1&, const M2& > &&
                                            ::std::is_assignable_v< M1&, LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& > > ) >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true > >
#endif
inline constexpr M1&
operator *= ( M1& m1, const M2& m2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& >, M1&, const M2& > &&
             ::std::is_assignable_v< M1&, LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& > > )
#endif
{
  return m1 = LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& >( m1, m2 );
}

LINALG_END // linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_MATRIX_PRODUCT_HPP
