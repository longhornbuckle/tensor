//==================================================================================================
//  File:       matrix_vector_product.hpp
//
//  Summary:    This header defines:
//              LINALG::accessor_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
//              LINALG::allocator_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
//              LINALG::layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
//              LINALG_EXPRESSIONS_DETAIL::vector_matrix_product_expression_traits< Vector, Matrix >
//              LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector >
//              LINALG::operator * ( const M& m, const V& v )
//==================================================================================================

#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_MATRIX_VECTOR_PRODUCT_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_MATRIX_VECTOR_PRODUCT_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//-------------------
//  Accessor Result
//-------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Matrix, class Vector >
struct accessor_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< Matrix >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< Matrix >::value_type >() * ::std::declval< typename ::std::remove_reference_t< Vector >::value_type >() ) >;
};

#else

template < class Matrix, class Vector, class Enable >
struct accessor_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector, Enable > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< Matrix >::accessor_type,
                                  decltype( ::std::declval< typename ::std::remove_reference_t< Matrix >::value_type >() * ::std::declval< typename ::std::remove_reference_t< Vector >::value_type >() ) >;
};

#endif

//--------------------
//  Allocator Result
//--------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Matrix, class Vector >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Matrix > > &&
             LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Vector > > )
struct allocator_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
{
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< Vector > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< Matrix > >,
                                                                Vector,
                                                                Matrix > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector >& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< Vector > > ||
                   ! LINALG_CONCEPTS::dynamic_tensor< ::std::remove_reference_t< Matrix > > )
    {
      return allocator_result< Vector >::get_allocator( t.second() );
    }
    else
    {
      return allocator_result< Matrix >::get_allocator( t.first() );
    }
  }
};

#else

template < class Matrix, class Vector, class Enable >
struct allocator_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector, Enable > >
{
private:
  using T = LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector, Enable >;
public:
  using type = typename allocator_result< ::std::conditional_t< LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Vector > > ||
                                                                  ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Matrix > >,
                                                                Vector,
                                                                Matrix > >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const T& t ) noexcept
  {
    if constexpr ( LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Vector > > ||
                    ! LINALG_CONCEPTS::dynamic_tensor_v< ::std::remove_reference_t< Matrix > > )
    {
      return allocator_result< Vector >::get_allocator( t.second() );
    }
    else
    {
      return allocator_result< Matrix >::get_allocator( t.first() );
    }
  }
};

#endif

//-----------------
//  Layout Result
//-----------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Matrix, class Vector >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Matrix > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Vector > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, ::std::layout_stride > ) &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Vector >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Vector >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Vector >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
{
  using type = ::std::conditional_t< ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, typename ::std::remove_reference_t< Vector >::layout_type > &&
                                       ! ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, ::std::layout_stride >,
                                     typename ::std::remove_reference_t< Matrix >::layout_type,
                                     LINALG::default_layout >;
};

template < class Matrix, class Vector >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Matrix > > &&
             LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Vector > > )
struct layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::matrix_vector_product_expression< decltype( ::std::declval< Matrix >().evaluate() ), Vector > > >::type;
};

template < class Matrix, class Vector >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Matrix > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Vector > > )
struct layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, decltype( ::std::declval< Vector >().evaluate() ) > > >::type;
};

template < class Matrix, class Vector >
  requires ( LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Matrix > > &&
             LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Vector > > )
struct layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector > >
{
  using type = typename layout_result< ::std::remove_reference_t< LINALG_EXPRESSIONS::matrix_vector_product_expression< decltype( ::std::declval< Matrix >().evaluate() ), decltype( ::std::declval< Vector >().evaluate() ) > > >::type;
};

#else

template < class Matrix, class Vector, class Enable >
struct layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< Matrix, Vector, Enable > >
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
    using type = typename layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< T, decltype( ::std::declval< U >().evaluate() ) > >::type;
  };
  template < class T, class U >
  struct unevaluated_readable_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< decltype( ::std::declval< T >().evaluate() ), U > >::type;
  };
  template < class T, class U >
  struct unevaluated_unevaluated_helper
  {
    using type = typename layout_result< LINALG_EXPRESSIONS::matrix_vector_product_expression< decltype( ::std::declval< T >().evaluate() ), decltype( ::std::declval< U >().evaluate() ) > >::type;
  };
public:
  using type = typename ::std::conditional_t< ( ( ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< Matrix >::layout_type, ::std::layout_stride > ) &&
                                                ( ::std::is_same_v< typename ::std::remove_reference_t< Vector >::layout_type, ::std::layout_right > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< Vector >::layout_type, ::std::layout_left > ||
                                                  ::std::is_same_v< typename ::std::remove_reference_t< Vector >::layout_type, ::std::layout_stride > ) ),
                                              typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Matrix > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Vector > >,
                                                                                          readable_readable_helper< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Vector > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > >,
                                                                                                                invalid_helper< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > > > >,
                                                                    ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Matrix > >,
                                                                                          ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Vector > >,
                                                                                                                readable_unevaluated_helper< ::std::remove_reference_t< Vector >, ::std::remove_reference_t< Matrix > >,
                                                                                                                ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Vector > >,
                                                                                                                                      unevaluated_unevaluated_helper< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > >,
                                                                                                                                      invalid_helper< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > > > >,
                                                                                          invalid_helper< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > > > >,
                                              invalid_helper< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > > >::type;
};

#endif

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

// Matrix Vector product expression traits
template < class Matrix, class Vector >
class matrix_vector_product_expression_traits
{
  public:
    // Aliases
    using value_type   = decltype( ::std::declval< typename ::std::remove_reference_t< Matrix >::value_type >() * ::std::declval< typename ::std::remove_reference_t< Vector >::value_type >() );
    using index_type   = ::std::common_type_t< typename ::std::remove_reference_t< Matrix >::index_type, typename ::std::remove_reference_t< Vector >::index_type >;
    using size_type    = ::std::common_type_t< typename ::std::remove_reference_t< Matrix >::size_type, typename ::std::remove_reference_t< Vector >::size_type >;
    using extents_type = ::std::extents< ::std::common_type_t< typename ::std::remove_reference_t< Matrix >::extents_type::index_type,
                                                               typename ::std::remove_reference_t< Vector >::extents_type::index_type >,
                                         ::std::remove_reference_t< Matrix >::extents_type::static_extent( 0 ) >;
    using rank_type    = ::std::common_type_t< typename ::std::remove_reference_t< Matrix >::rank_type, typename ::std::remove_reference_t< Vector >::rank_type >;
};

LINALG_EXPRESSIONS_DETAIL_END // expression detail namespace

LINALG_EXPRESSIONS_BEGIN // expressions namespace

#ifdef LINALG_ENABLE_CONCEPTS
template < class Matrix, class Vector >
  requires ( LINALG_CONCEPTS::matrix_expression< ::std::remove_reference_t< Matrix > > &&
             LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< Vector > > &&
             ( requires ( const typename ::std::remove_reference_t< Matrix >::value_type& v1, const typename ::std::remove_reference_t< Vector >::value_type& v2 ) { v1 * v2; } ) &&
             ( ( ::std::remove_reference_t< Vector >::extents_type::static_extent(0) == ::std::remove_reference_t< Matrix >::extents_type::static_extent(1) ) ||
               ( ::std::remove_reference_t< Vector >::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
               ( ::std::remove_reference_t< Matrix >::extents_type::static_extent(1) == ::std::dynamic_extent ) ) )
class matrix_vector_product_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< matrix_vector_product_expression< Matrix, Vector >,
                                                                   LINALG_EXPRESSIONS_DETAIL::matrix_vector_product_expression_traits< Matrix, Vector > >
#else
template < class Matrix, class Vector, typename Enable >
class matrix_vector_product_expression :
  public LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< matrix_vector_product_expression< Matrix, Vector, Enable >,
                                                                   LINALG_EXPRESSIONS_DETAIL::matrix_vector_product_expression_traits< Matrix, Vector > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = matrix_vector_product_expression< Matrix, Vector >;
    #else
    using self_type   = matrix_vector_product_expression< Matrix, Vector, Enable >;
    #endif
    using traits_type = LINALG_EXPRESSIONS_DETAIL::matrix_vector_product_expression_traits< Matrix, Vector >;
    using base_type   = LINALG_EXPRESSIONS_DETAIL::binary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr matrix_vector_product_expression( Matrix&& m, Vector&& v )
      noexcept( ( ::std::remove_reference_t< Vector >::extents_type::static_extent( 0 ) == ::std::remove_reference_t< Matrix >::extents_type::static_extent( 1 ) ) &&
                ( ::std::remove_reference_t< Vector >::extents_type::static_extent( 0 ) != ::std::dynamic_extent ) ) :
      m_(m), v_(v)
    {
      if constexpr ( !( ( ::std::remove_reference_t< Vector >::extents_type::static_extent( 0 ) == ::std::remove_reference_t< Matrix >::extents_type::static_extent( 1 ) ) &&
                        ( ::std::remove_reference_t< Vector >::extents_type::static_extent( 0 ) != ::std::dynamic_extent ) ) )
      {
        using common_index_type = ::std::common_type_t< typename ::std::remove_reference_t< Matrix >::index_type,
                                                        typename ::std::remove_reference_t< Vector >::index_type >;
        if ( static_cast< common_index_type >( v.extent( 0 ) ) != static_cast< common_index_type >( m.extent( 1 ) ) ) LINALG_UNLIKELY
        {
          throw ::std::length_error( "Matrix Vector extents are incompatable." );
        }
      }
    }
    constexpr matrix_vector_product_expression& operator = ( const matrix_vector_product_expression& t ) noexcept { this->v_ = t.v_; this->m_ = t.m_; }
    constexpr matrix_vector_product_expression& operator = ( matrix_vector_product_expression&& t ) noexcept { this->v_ = t.v_; this->m_ = t.m_; }
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
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< Vector >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return extents_type( this->m_.extent(0) ); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { return this->m_.extent( n ); }
    // Binary tensor expression function
    [[nodiscard]] constexpr const Matrix& first() const noexcept { return this->m_; }
    [[nodiscard]] constexpr const Vector& second() const noexcept { return this->v_; }
  private:
    // Access
    [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION value_type access( index_type index ) const
      noexcept( noexcept( LINALG_DETAIL::access( ::std::declval< Matrix >(), index, ::std::declval< index_type >() ) ) &&
                noexcept( LINALG_DETAIL::access( ::std::declval< Vector >(), ::std::declval< index_type >() ) ) )
    {
      value_type val { 0 };
      if constexpr ( ::std::remove_reference_t< Vector >::static_extent(0) != ::std::dynamic_extent )
      {
        for ( typename ::std::remove_reference_t< Vector >::index_type count = 0; count < this->v_.extent(0); ++count )
        {
          val += LINALG_DETAIL::access( this->m_, index, count ) * LINALG_DETAIL::access( this->v_, count );
        }
      }
      else
      {
        for ( typename ::std::remove_reference_t< Matrix >::index_type count = 0; count < this->m_.extent(1); ++count )
        {
          val += LINALG_DETAIL::access( this->m_, index, count ) * LINALG_DETAIL::access( this->v_, count );
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
    // Matrix Vector product
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
    Matrix& m_;
    Vector& v_;
};

LINALG_EXPRESSIONS_END // end expressions namespace

LINALG_BEGIN // linalg namespace

//-----------------------------
//  Matrix Vector product operators
//-----------------------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class M, class V >
#else
template < class M, class V,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_vector_product_expression< const M&, const V& >, const M&, const V& > >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const M& m, const V& v ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_vector_product_expression< const M&, const V& >, const M&, const V& >
#endif
{
  return LINALG_EXPRESSIONS::matrix_vector_product_expression< const M&, const V& >( m, v );
}

LINALG_END // linalg namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_VECTOR_MATRIX_PRODUCT_HPP
