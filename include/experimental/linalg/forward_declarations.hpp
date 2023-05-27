//==================================================================================================
//  File:       forward_declarations.hpp
//
//  Summary:    This header forward declares the primary linear algebra classes.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP
#define LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP

#include <experimental/linear_algebra.hpp>

// namespaces
// LINALG
LINALG_BEGIN LINALG_END
// LINALG_DETAIL
LINALG_DETAIL_BEGIN LINALG_DETAIL_END
// LINALG_CONCEPTS
LINALG_CONCEPTS_BEGIN LINALG_CONCEPTS_END
// LINALG_EXPRESSIONS
LINALG_EXPRESSIONS_BEGIN LINALG_EXPRESSIONS_END

// Default layout
LINALG_BEGIN
using default_layout = ::std::experimental::layout_right;
LINALG_END

//-  Tensor Expressions

LINALG_EXPRESSIONS_BEGIN // expressions namespace

// Unary Tensor Expressions

// Negate
#ifdef LINALG_ENABLE_CONCEPTS
template < tensor_expression Tensor >
#else
template < class Tensor, typename Enable = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
class negate_tensor_expression;

// Transpose indices
template < ::std::size_t index1 = 0, ::std::size_t index2 = 1 >
struct transpose_indices_t;
struct transpose_indices_v;

// Transpose
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor, class Transpose = transpose_indices_t<> >
#else
template < class Tensor,
           class Transpose = transpose_indices_t<>,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > &&
                                          !LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
class transpose_tensor_expression;

// Conjugate
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor, class Transpose = transpose_indices_t<> >
#else
template < class Tensor,
           class Transpose = transpose_indices_t<>,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > &&
                                          !LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
class conjugate_tensor_expression;

// Binary Tensor Expression

// Addition
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression FirstTensor, LINALG_CONCEPTS::tensor_expression SecondTensor >
#else
template < class FirstTensor, class SecondTensor,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< FirstTensor > &&
                                          LINALG_CONCEPTS::tensor_expression_v< SecondTensor > &&
                                          LINALG_CONCEPTS::has_equal_ranks_v< FirstTensor, SecondTensor > &&
                                          LINALG_CONCEPTS::may_have_equal_extents_v< FirstTensor, SecondTensor > &&
                                          LINALG_CONCEPTS::elements_are_additive_v< FirstTensor, SecondTensor > > >
#endif
class addition_tensor_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( FirstTensor::rank() == SecondTensor::rank() ) &&
             LINALG_DETAIL::extents_maybe_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > &&
             requires ( typename FirstTensor::value_type v1, typename SecondTensor::value_type v2 ) { v1 + v2 }; )
#endif
;

// Subtraction
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression FirstTensor, LINALG_CONCEPTS::tensor_expression SecondTensor >
#else
template < class FirstTensor, class SecondTensor,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< FirstTensor > &&
                                          LINALG_CONCEPTS::tensor_expression_v< SecondTensor > &&
                                          LINALG_CONCEPTS::has_equal_ranks_v< FirstTensor, SecondTensor > &&
                                          LINALG_CONCEPTS::may_have_equal_extents_v< FirstTensor, SecondTensor > &&
                                          LINALG_CONCEPTS::elements_are_subtractive_v< FirstTensor, SecondTensor > > >
#endif
class subtraction_tensor_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( FirstTensor::rank() == SecondTensor::rank() ) &&
             LINALG_DETAIL::extents_maybe_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > &&
             requires ( typename FirstTensor::value_type v1, typename SecondTensor::value_type v2 ) { v1 - v2 }; )
#endif
;

// Scalar Pre-Multiply
#ifdef LINALG_ENABLE_CONCEPTS
template < class S, LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class S, class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_is_scalar_premultiplicative_v< S, Tensor > > >
#endif
class scalar_preprod_tensor_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires ( const S& s, const typename Tensor::value_type& v ) { { s * v; } }
#endif
;

// Scalar Post-Multiply
#ifdef LINALG_ENABLE_CONCEPTS
template < class S, LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class S, class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_is_scalar_postmultiplicative_v< S, Tensor > > >
#endif
class scalar_postprod_tensor_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires ( const S& s, const typename Tensor::value_type& v ) { { v * s; } }
#endif
;

// Scalar Division
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor, class S >
#else
template < class Tensor, class S, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_is_scalar_divisible_v< S, Tensor > > >
#endif
class scalar_division_tensor_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires ( const S& s, const typename Tensor::value_type& v ) { { v / s; } }
#endif
;

// Scalar Modulo
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor, class S >
#else
template < class Tensor, class S, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_is_scalar_modulo_v< S, Tensor > > >
#endif
class scalar_modulo_tensor_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires ( const S& s, const typename Tensor::value_type& v ) { { v % s; } }
#endif
;

// Matrix Product
#ifdef LINALG_ENABLE_CONCEPTS
template < matrix_expression FirstMatrix, matrix_expression SecondMatrix >
#else
template < class FirstMatrix, class SecondMatrix, typename = ::std::enable_if_t< LINALG_CONCEPTS::matrix_expression_v< FirstMatrix > &&
                                                                                 LINALG_CONCEPTS::matrix_expression_v< SecondMatrix > &&
                                                                                 LINALG_CONCEPTS::elements_are_multiplicative_v< FirstMatrix, SecondMatrix > &&
                                                                                 LINALG_CONCEPTS::matrices_may_be_multiplicative_v< FirstMatrix, SecondMatrix > > >
#endif
class matrix_product_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( requires ( const typename FirstMatrix::value_type& v1, const typename SecondMatrix::value_type& v2 ) { { v1 * v2; } } ) &&
             ( ( FirstMatrix::extents_type::static_extent(1) == SecondMatrix::extents_type::static_extent(0) ) ||
               ( FirstMatrix::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ||
               ( SecondMatrix::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ) )
#endif
;

// Vector Matrix Product
#ifdef LINALG_ENABLE_CONCEPTS
template < vector_expression Vector, matrix_expression Matrix >
#else
template < class Vector, class Matrix, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< Vector > &&
                                                                      LINALG_CONCEPTS::matrix_expression_v< Matrix > &&
                                                                      LINALG_CONCEPTS::elements_are_multiplicative_v< Vector, Matrix > &&
                                                                      LINALG_CONCEPTS::vector_matrix_may_be_multiplicative_v< Vector, Matrix > > >
#endif
class vector_matrix_product_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( requires ( const typename Vector::value_type& v1, const typename Matrix::value_type& v2 ) { { v1 * v2; } } ) &&
             ( ( Vector::extents_type::static_extent(0) == Matrix::extents_type::static_extent(0) ) ||
               ( Vector::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ||
               ( Matrix::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ) )
#endif
;

// Matrix Vector Product
#ifdef LINALG_ENABLE_CONCEPTS
template < matrix_expression Matrix, vector_expression Vector >
#else
template < class Matrix, class Vector, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< Vector > &&
                                                                      LINALG_CONCEPTS::matrix_expression_v< Matrix > &&
                                                                      LINALG_CONCEPTS::elements_are_multiplicative_v< Vector, Matrix > &&
                                                                      LINALG_CONCEPTS::matrix_vector_may_be_multiplicative_v< Matrix, Vector > > >
#endif
class matrix_vector_product_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( requires ( const typename Matrix::value_type& v1, const typename Vector::value_type& v2 ) { { v1 * v2; } } ) &&
             ( ( Vector::extents_type::static_extent(0) == Matrix::extents_type::static_extent(1) ) ||
               ( Vector::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
               ( Matrix::extents_type::static_extent(1) == ::std::dynamic_extent ) ) )
#endif
;

// Outer Product
#ifdef LINALG_ENABLE_CONCEPTS
template < vector_expression FirstVector, vector_expression SecondVector >
#else
template < class FirstVector, class SecondVector, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< FirstVector > &&
                                                                                 LINALG_CONCEPTS::vector_expression_v< SecondVector > &&
                                                                                 LINALG_CONCEPTS::elements_are_multiplicative_v< FirstVector, SecondVector > > >
#endif
class outer_product_expression
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( requires ( const typename FirstVector::value_type& v1, const typename SecondVector::value_type& v2 ) { { v1 * v2; } } )
#endif
;

LINALG_EXPRESSIONS_END // end expressions namespace

LINALG_BEGIN // linalg namespace

// Fixed Size Tensor
template < class T,
           class Extents,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::experimental::default_accessor< T > >
class fs_tensor
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( Extents::rank_dynamic() == 0 )
#endif
;

// Dynamic Resizable Tensor
template < class T,
           class Extents,
           class LayoutPolicy   = default_layout,
           class CapExtents     = Extents,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor< T > >
class dr_tensor;

// Dynamic Tensor
template < class         T,
           ::std::size_t N,
           class LayoutPolicy   = default_layout,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor< T > >
using dyn_tensor = dr_tensor< T,
                              LINALG_DETAIL::dyn_extents< ::std::size_t, N >,
                              LayoutPolicy,
                              LINALG_DETAIL::dyn_extents< ::std::size_t, N >,
                              Allocator,
                              AccessorPolicy >;



// Alias mdspan
template < class ElementType,
           class Extents,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::experimental::default_accessor<ElementType> >
using tensor_view = ::std::experimental::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >;

// Alias matrix view
template < class ElementType,
           auto  R,
           auto  C,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::experimental::default_accessor<ElementType> >
using matrix_view = tensor_view< ElementType, ::std::experimental::extents<::std::common_type_t<decltype(R),decltype(C)>,static_cast<::std::size_t>(R),static_cast<::std::size_t>(C)>, LayoutPolicy, AccessorPolicy >;

// Alias matrix view
template < class ElementType,
           auto  N,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::experimental::default_accessor<ElementType> >
using vector_view = tensor_view< ElementType, ::std::experimental::extents<decltype(N),static_cast<::std::size_t>(N)>, LayoutPolicy, AccessorPolicy >;

// Alias for dr_matrix
template < class T,
           auto  R,
           auto  C,
           class LayoutPolicy   = default_layout,
           auto  Rc             = R,
           auto  Cc             = C,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
using dr_matrix = dr_tensor< T, ::std::experimental::extents<::std::common_type_t<decltype(R),decltype(C)>,static_cast<::std::size_t>(R),static_cast<::std::size_t>(C)>, LayoutPolicy, ::std::experimental::extents<::std::common_type_t<decltype(Rc),decltype(Cc)>,static_cast<::std::size_t>(Rc),static_cast<::std::size_t>(Cc)>, Allocator, AccessorPolicy >;

// Alias for dr_vector
template < class T,
           auto  N,
           class LayoutPolicy   = default_layout,
           auto  Nc             = N,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
using dr_vector = dr_tensor< T, ::std::experimental::extents<decltype(N),static_cast<::std::size_t>(N)>, LayoutPolicy, ::std::experimental::extents<decltype(Nc),static_cast<::std::size_t>(Nc)>, Allocator, AccessorPolicy >;

// Alias for fs_matrix
template < class T,
           auto  R,
           auto  C,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
using fs_matrix = fs_tensor< T, ::std::experimental::extents<::std::common_type_t<decltype(R),decltype(C)>,static_cast<::std::size_t>(R),static_cast<::std::size_t>(C)>, LayoutPolicy, AccessorPolicy >;

// Alias for fs_vector
template < class T,
           auto  N,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
using fs_vector = fs_tensor< T, ::std::experimental::extents<decltype(N),static_cast<::std::size_t>(N)>, LayoutPolicy, AccessorPolicy >;

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP
