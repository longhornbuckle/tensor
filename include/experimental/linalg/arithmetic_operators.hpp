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

// //=================================================================================================
// //  Vector Matrix product
// //=================================================================================================
// #ifdef LINALG_ENABLE_CONCEPTS
// template < class V, class M >
// #else
// template < class V, class M,
//            typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< const V&, const M& >, const V&, const M& > >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true > >
// #endif
// [[nodiscard]] inline constexpr decltype(auto)
// operator * ( const V& v, const M& m ) noexcept
// #ifdef LINALG_ENABLE_CONCEPTS
//   requires ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< const V&, const M& >, const V&, const M& >
// #endif
// {
//   return LINALG_EXPRESSIONS::vector_matrix_product_expression( v, m );
// }

// //=================================================================================================
// //  Vector Matrix multiplication assignment
// //=================================================================================================
// #ifdef LINALG_ENABLE_CONCEPTS
// template < class V, class M >
// #else
// template < class V, class M,
//            typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& >, V&, const M& > &&
//                                             ::std::is_assignable_v< V&, LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& > > ) >,
//            typename = ::std::enable_if_t< true > >
// #endif
// [[nodiscard]] inline constexpr V&
// operator *= ( V& v, const M& m ) noexcept
// #ifdef LINALG_ENABLE_CONCEPTS
//   requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& >, V&, const M& > &&
//              ::std::is_assignable_v< V&, LINALG_EXPRESSIONS::vector_matrix_product_expression< V&, const M& > > )
// #endif
// {
//   return v = LINALG_EXPRESSIONS::vector_matrix_product_expression( v, m );
// }

// //=================================================================================================
// //  Matrix Vector product
// //=================================================================================================
// #ifdef LINALG_ENABLE_CONCEPTS
// template < class M, class V >
// #else
// template < class M, class V,
//            typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_vector_product_expression< const M&, const V& >, const M&, const V& > >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true > >
// #endif
// [[nodiscard]] inline constexpr decltype(auto)
// operator * ( const M& m, const V& v ) noexcept
// #ifdef LINALG_ENABLE_CONCEPTS
//   requires ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_vector_product_expression< const M&, const V& >, const M&, const V& >
// #endif
// {
//   return LINALG_EXPRESSIONS::matrix_vector_product_expression( m, v );
// }

// //=================================================================================================
// //  Matrix Matrix product
// //=================================================================================================
// #ifdef LINALG_ENABLE_CONCEPTS
// template < class M1, class M2 >
// #else
// template < class M1, class M2,
//            typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< const M1&, const M2& >, const M1&, const M2& > >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true > >
// #endif
// [[nodiscard]] inline constexpr decltype(auto)
// operator * ( const M1& m1, const M2& m2 ) noexcept
// #ifdef LINALG_ENABLE_CONCEPTS
//   requires ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< const M1&, const M2& >, const M1&, const M2& >
// #endif
// {
//   return LINALG_EXPRESSIONS::matrix_vector_product_expression( m1, m2 );
// }

// //=================================================================================================
// //  Matrix multiplication assignment
// //=================================================================================================
// #ifdef LINALG_ENABLE_CONCEPTS
// template < class M1, class M2 >
// #else
// template < class M1, class M2,
//            typename = ::std::enable_if_t< ( ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& >, M1&, const M2& > &&
//                                             ::std::is_assignable_v< M1&, LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& > > ) >,
//            typename = ::std::enable_if_t< true >,
//            typename = ::std::enable_if_t< true > >
// #endif
// [[nodiscard]] inline constexpr M1&
// operator *= ( M1& m1, const M2& m2 ) noexcept
// #ifdef LINALG_ENABLE_CONCEPTS
//   requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& >, M1&, const M2& > &&
//              ::std::is_assignable_v< M1&, LINALG_EXPRESSIONS::matrix_product_expression< M1&, const M2& > > )
// #endif
// {
//   return m1 = LINALG_EXPRESSIONS::matrix_product_expression( m1, m2 );
// }

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
