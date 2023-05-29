//==================================================================================================
//  File:       arithmetic_operators.hpp
//
//  Summary:    This header defines the overloaded operators that implement basic arithmetic
//              operations on vectors, matrices, and tensors.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP
#define LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//=================================================================================================
//  Unary negation operator
//=================================================================================================
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

//=================================================================================================
//  Unary transpose operator
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T >
#else
template < class T,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t<> >,
                                                                                                                      const T&,
                                                                                                                      LINALG_EXPRESSIONS::transpose_indices_t<> > &&
                                          ( LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< T > > ||
                                            LINALG_CONCEPTS::matrix_expression_v< ::std::remove_reference_t< T > > ) > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const T& t ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t<> >,
                                                                                         const T&,
                                                                                         LINALG_EXPRESSIONS::transpose_indices_t<> > &&
             ( ::std::remove_reference_t< T >::rank() < 3 ) )
#endif
{
  return LINALG_EXPRESSIONS::transpose_tensor_expression< const T& >( t, LINALG_EXPRESSIONS::transpose_indices_t<> {} );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, auto index1, auto index2 >
#else
template < class T, auto index1, auto index2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >,
                                                                                                                      const T&,
                                                                                                                      LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const T& t, const LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 >& indices = LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > {} ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >,
                                                                                       const T&,
                                                                                       LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >
#endif
{
  return LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >( t, indices );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class IndexType >
#else
template < class T, class IndexType,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > >,
                                                                     const T&,
                                                                     LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const T& t, IndexType index1, IndexType index2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > >,
                                                                                       const T&,
                                                                                       LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > >
#endif
{
  return LINALG_EXPRESSIONS::transpose_tensor_expression< const T&,LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > >
    ( t, LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > { index1, index2 } );
}

//=================================================================================================
//  Unary conjugate transpose operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T >
#else
template < class T,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t<> >,
                                                                                                                      const T&,
                                                                                                                      LINALG_EXPRESSIONS::transpose_indices_t<> > &&
                                          ( LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< T > > ||
                                            LINALG_CONCEPTS::matrix_expression_v< ::std::remove_reference_t< T > > ) > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
conj( const T& t ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t<> >,
                                                                                         const T&,
                                                                                         LINALG_EXPRESSIONS::transpose_indices_t<> > &&
             ( ::std::remove_reference_t< T >::rank() < 3 ) )
#endif
{
  return LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t<> >( t, LINALG_EXPRESSIONS::transpose_indices_t<> {} );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, auto index1, auto index2 >
#else
template < class T, auto index1, auto index2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >,
                                                                                                                      const T&,
                                                                                                                      LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
conj( const T& t, const LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 >& indices = LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > {} ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >,
                                                                                       const T&,
                                                                                       LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >
#endif
{
  return LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t< index1, index2 > >( t, indices );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class IndexType >
#else
template < class T, class IndexType,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > >,
                                                                     const T&,
                                                                     LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
conj( const T& t, IndexType index1, IndexType index2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > >,
                                                                                       const T&,
                                                                                       LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > >
#endif
{
  return LINALG_EXPRESSIONS::conjugate_tensor_expression< const T&, LINALG_EXPRESSIONS::transpose_indices_t<> >( t, LINALG_EXPRESSIONS::transpose_indices_v< IndexType, IndexType > { index1, index2 } );
}

//=================================================================================================
//  Binary addition operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T1, class T2 >
#else
template < class T1, class T2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< const T1&, const T2& >, const T1&, const T2& > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator + ( const T1& t1, const T2& t2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< const T1&, const T2& >, const T1&, const T2& >
#endif
{
  return LINALG_EXPRESSIONS::addition_tensor_expression( t1, t2 );
}

//=================================================================================================
//  Binary addition assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T1, class T2 >
#else
template < class T1, class T2,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& >, T1&, const T2& > &&
                                            ::std::is_assignable_v< T1&, LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& > > ) > >
#endif
inline constexpr T1&
operator += ( T1& t1, const T2& t2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& >, T1&, const T2& > &&
             ::std::is_assignable_v< T1&, LINALG_EXPRESSIONS::addition_tensor_expression< T1&, const T2& > > )
#endif
{
  return t1 = LINALG_EXPRESSIONS::addition_tensor_expression( t1, t2 );
}

//=================================================================================================
//  Binary subtraction operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T1, class T2 >
#else
template < class T1, class T2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::subtraction_tensor_expression< const T1&, const T2& >, const T1&, const T2& > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator - ( const T1& t1, const T2& t2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::subtraction_tensor_expression< const T1&, const T2& >, const T1&, const T2& >
#endif
{
  return LINALG_EXPRESSIONS::subtraction_tensor_expression( t1, t2 );
}

//=================================================================================================
//  Binary subtraction assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T1, class T2 >
#else
template < class T1, class T2,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::subtraction_tensor_expression< T1&, const T2& >, T1&, const T2& > &&
                                            ::std::is_assignable_v< T1&, LINALG_EXPRESSIONS::subtraction_tensor_expression< T1&, const T2& > > ) > >
#endif
inline constexpr T1&
operator -= ( T1& t1, const T2& t2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::subtraction_tensor_expression< T1&, const T2& >, T1&, const T2& > &&
             ::std::is_assignable_v< T1&, LINALG_EXPRESSIONS::subtraction_tensor_expression< T1&, const T2& > > )
#endif
{
  return t1 = LINALG_EXPRESSIONS::subtraction_tensor_expression( t1, t2 );
}

//=================================================================================================
//  Scalar pre-multiplication operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class S, class T >
#else
template < class S, class T,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_preprod_tensor_expression< const S&, const T& >, const S&, const T& > >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const S& s, const T& t ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_preprod_tensor_expression< const S&, const T& >, const S&, const T& >
#endif
{
  return LINALG_EXPRESSIONS::scalar_preprod_tensor_expression( s, t );
}

//=================================================================================================
//  Scalar post-multiplication operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_postprod_tensor_expression< const T&, const S& >, const T&, const S& > >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_postprod_tensor_expression< const T&, const S& >, const T&, const S& >
#endif
{
  return LINALG_EXPRESSIONS::scalar_postprod_tensor_expression( t, s );
}

//=================================================================================================
//  Scalar post-multiplication assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_postprod_tensor_expression< T&, const S& >, T&, const S& > &&
                                            ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_postprod_tensor_expression< T&, const S& > > ) > >
#endif
[[nodiscard]] inline constexpr T&
operator *= ( T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_postprod_tensor_expression< T&, const S& >, T&, const S& > &&
             ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_postprod_tensor_expression< T&, const S& > > )
#endif
{
  return t = LINALG_EXPRESSIONS::scalar_postprod_tensor_expression( t, s );
}

//=================================================================================================
//  Scalar divison operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_division_tensor_expression< const T&, const S& >, const T&, const S& > >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator / ( const T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_division_tensor_expression< const T&, const S& >, const T&, const S& >
#endif
{
  return LINALG_EXPRESSIONS::scalar_division_tensor_expression( t, s );
}

//=================================================================================================
//  Scalar divison assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_division_tensor_expression< T&, const S& >, T&, const S& > &&
                                            ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_division_tensor_expression< T&, const S& > > ) > >
#endif
[[nodiscard]] inline constexpr T&
operator /= ( T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_division_tensor_expression< T&, const S& >, T&, const S& > &&
             ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_division_tensor_expression< T&, const S& > > )
#endif
{
  return t = LINALG_EXPRESSIONS::scalar_division_tensor_expression( t, s );
}

//=================================================================================================
//  Scalar modulo operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< const T&, const S& >, const T&, const S& > >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator % ( const T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< const T&, const S& >, const T&, const S& >
#endif
{
  return LINALG_EXPRESSIONS::scalar_modulo_tensor_expression( t, s );
}

//=================================================================================================
//  Scalar modulo assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class S >
#else
template < class T, class S,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& >, T&, const S& > &&
                                            ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& > > ) > >
#endif
[[nodiscard]] inline constexpr T&
operator %= ( T& t, const S& s ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& >, T&, const S& > &&
             ::std::is_assignable_v< T&, LINALG_EXPRESSIONS::scalar_modulo_tensor_expression< T&, const S& > > )
#endif
{
  return t = LINALG_EXPRESSIONS::scalar_modulo_tensor_expression( t, s );
}

//=================================================================================================
//  Inner product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::vector_expression V1, LINALG_CONCEPTS::vector_expression V2 >
#else
template < class V1, class V2, typename = ::std::enable_if_t< ( ( V1::extents_type::static_extent(0) == V2::extents_type::static_extent(0) ) ||
                                                                ( V1::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
                                                                ( V2::extents_type::static_extent(0) == ::std::dynamic_extent ) ) &&
                                                              ( LINALG_CONCEPTS::elements_are_multiplicative_v< V1, V2 > ) > >
#endif
[[nodiscard]] inline constexpr auto
inner_prod( const V1& v1, const V2 v2 )
  noexcept( LINALG_DETAIL::extents_are_equal_v< typename V1::extents_type, typename V2::extents_type > &&
            noexcept( LINALG_DETAIL::access( v1, ::std::declval< typename V1::index_type >() ) ) &&
            noexcept( LINALG_DETAIL::access( v2, ::std::declval< typename V2::index_type >() ) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( V1::extents_type::static_extent(0) == V2::extents_type::static_extent(0) ) ||
             ( V1::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
             ( V2::extents_type::static_extent(0) == ::std::dynamic_extent ) ) &&
           requires ( const typename V1::value_type& val1, const typename V2::value_type& val2 ) { { val1 * val2 }; }
#endif
{
  if constexpr ( !LINALG_DETAIL::extents_are_equal_v< typename V1::extents_type, typename V2::extents_type > )
  {
    if ( v1.extent(0) != v2.extent(0) ) LINALG_UNLIKELY
    {
      throw length_error( "Vector extents are incompatable." );
    }
  }
  decltype( ::std::declval< typename V1::value_type >() * ::std::declval< typename V2::value_type >() )
    val { 0 };
  if constexpr ( V1::extents_type::static_extent(0) != ::std::dynamic_extent )
  {
    for ( auto count = 0; count < v1.extent(0); ++count )
    {
      val += LINALG_DETAIL::access( v1, count ) * LINALG_DETAIL::access( v2, count );
    }
  }
  else
  {
    for ( auto count = 0; count < v2.extent(0); ++count )
    {
      val += LINALG_DETAIL::access( v1, count ) * LINALG_DETAIL::access( v2, count );
    }
  }
  return val;
}

//=================================================================================================
//  Outer product
//=================================================================================================
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
  return LINALG_EXPRESSIONS::outer_product_expression( v1, v2 );
}

//=================================================================================================
//  Vector Matrix product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class V, class M >
#else
template < class V, class M,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< const V&, const M& >, const V&, const M& > >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const V& v, const M& m ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< const V&, const M& >, const V&, const M& >
#endif
{
  return LINALG_EXPRESSIONS::vector_matrix_product_expression( v, m );
}

//=================================================================================================
//  Vector Matrix multiplication assignment
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class V, class M >
#else
template < class V, class M,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& >, V&, const M& > &&
                                            ::std::is_assignable_v< V&, LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& > > ) >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr V&
operator *= ( V& v, const M& m ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& >, V&, const M& > &&
             ::std::is_assignable_v< V&, LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& > > )
#endif
{
  return v = LINALG_EXPRESSIONS::vector_matrix_product_expression( v, m );
}

//=================================================================================================
//  Matrix Vector product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class M, class V >
#else
template < class M, class V,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_vector_product_expression< const M&, const V& >, const M&, const V& > >,
           typename = ::std::enable_if_t< true >,
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
  return LINALG_EXPRESSIONS::matrix_vector_product_expression( m, v );
}

//=================================================================================================
//  Matrix Matrix product
//=================================================================================================
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
  return LINALG_EXPRESSIONS::matrix_vector_product_expression( m1, m2 );
}

//=================================================================================================
//  Matrix multiplication assignment
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class M1, class M2 >
#else
template < class M1, class M2,
           typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& >, M1&, const M2& > &&
                                            ::std::is_assignable_v< M1&, LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& > > ) >,
           typename = ::std::enable_if_t< true >,
           typename = ::std::enable_if_t< true > >
#endif
[[nodiscard]] inline constexpr M1&
operator *= ( M1& m1, const M2& m2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& >, M1&, const M2& > &&
             ::std::is_assignable_v< M1&, LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& > > )
#endif
{
  return m1 = LINALG_EXPRESSIONS::matrix_product_expression( m1, m2 );
}

LINALG_END // end linalg namespace

// Bring operators into std namespace for use with mdspan
namespace  std
{
  using LINALG::operator -;
  using LINALG::operator +;
  using LINALG::operator *;
  using LINALG::operator /;
  using LINALG::operator %;
  using LINALG::operator -=;
  using LINALG::operator +=;
  using LINALG::operator *=;
  using LINALG::operator /=;
  using LINALG::operator %=;
  using LINALG::trans;
  using LINALG::conj;
  using LINALG::inner_prod;
  using LINALG::outer_prod;
} // std namespace

#endif  //- LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP
