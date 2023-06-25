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
// LINALG_EXPRESSIONS_DETAIL
LINALG_EXPRESSIONS_DETAIL_BEGIN LINALG_EXPRESSIONS_DETAIL_END

// Default layout
LINALG_BEGIN
using default_layout = ::std::layout_right;
LINALG_END

//-  Tensor Expressions

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions detail namespace

template < class Tensor, class Traits >
class unary_tensor_expression_base;

template < class Tensor, class Traits >
class binary_tensor_expression_base;

// Transpose indices
template < ::std::size_t index1 = 0, ::std::size_t index2 = 1 >
struct transpose_indices_t;
template < class IndexType1 = ::std::size_t, class IndexType2 = ::std::size_t >
struct transpose_indices_v;

LINALG_EXPRESSIONS_DETAIL_END // expressions detail namesapce

LINALG_EXPRESSIONS_BEGIN // expressions namespace

// Unary Tensor Expressions

// Negate
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
#else
template < class Tensor, typename Enable = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
class negate_tensor_expression;

// Transpose
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose = LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<> >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
#else
template < class Tensor,
           class Transpose = LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<>,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
class transpose_tensor_expression;

// Conjugate
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose = LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<> >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
#else
template < class Tensor,
           class Transpose = LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<>,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > > >
#endif
class conjugate_tensor_expression;

// Binary Tensor Expression

// Addition
#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondTensor > > &&
             ( ::std::remove_reference_t< FirstTensor >::rank() == ::std::remove_reference_t< SecondTensor >::rank() ) &&
             LINALG_DETAIL::extents_may_be_equal_v< typename ::std::remove_reference_t< FirstTensor >::extents_type, typename ::std::remove_reference_t< SecondTensor >::extents_type > &&
             requires ( typename ::std::remove_reference_t< FirstTensor >::value_type v1, typename ::std::remove_reference_t< SecondTensor >::value_type v2 ) { v1 + v2; } )
#else
template < class FirstTensor, class SecondTensor,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< FirstTensor > > &&
                                          LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< SecondTensor > > &&
                                          LINALG_CONCEPTS::has_equal_ranks_v< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > &&
                                          LINALG_CONCEPTS::may_have_equal_extents_v< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > &&
                                          LINALG_CONCEPTS::elements_are_additive_v< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > > >
#endif
class addition_tensor_expression;

// Subtraction
#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstTensor, class SecondTensor >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< FirstTensor > > &&
             LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< SecondTensor > > &&
             ( ::std::remove_reference_t< FirstTensor >::rank() == ::std::remove_reference_t< SecondTensor >::rank() ) &&
             LINALG_DETAIL::extents_may_be_equal_v< typename ::std::remove_reference_t< FirstTensor >::extents_type, typename ::std::remove_reference_t< SecondTensor >::extents_type > &&
             requires ( typename ::std::remove_reference_t< FirstTensor >::value_type v1, typename ::std::remove_reference_t< SecondTensor >::value_type v2 ) { v1 - v2; } )
#else
template < class FirstTensor, class SecondTensor,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< FirstTensor > > &&
                                          LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< SecondTensor > > &&
                                          LINALG_CONCEPTS::has_equal_ranks_v< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > &&
                                          LINALG_CONCEPTS::may_have_equal_extents_v< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > &&
                                          LINALG_CONCEPTS::elements_are_subtractive_v< ::std::remove_reference_t< FirstTensor >, ::std::remove_reference_t< SecondTensor > > > >
#endif
class subtraction_tensor_expression;

// Scalar Pre-Multiply
#ifdef LINALG_ENABLE_CONCEPTS
template < class Scalar, class Tensor >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > > &&
             ! LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Scalar > > &&
             requires ( Scalar s, typename ::std::remove_reference_t< Tensor >::value_type v ) { s * v; } )
#else
template < class Scalar, class Tensor,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > &&
                                          ! LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Scalar > > &&
                                          LINALG_CONCEPTS::tensor_is_scalar_premultiplicative_v< ::std::remove_reference_t< Scalar >, ::std::remove_reference_t< Tensor > > > >
#endif
class scalar_preprod_tensor_expression;

// Scalar Post-Multiply
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Scalar >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > > &&
             ! LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Scalar > > &&
             requires ( typename ::std::remove_reference_t< Tensor >::value_type v, Scalar s ) { v * s; } )
#else
template < class Tensor, class Scalar,
           typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Tensor > > &&
                                          ! LINALG_CONCEPTS::tensor_expression_v< ::std::remove_reference_t< Scalar > > &&
                                          LINALG_CONCEPTS::tensor_is_scalar_postmultiplicative_v< ::std::remove_reference_t< Tensor >, ::std::remove_reference_t< Scalar > > > >
#endif
class scalar_postprod_tensor_expression;

// Scalar Division
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Scalar >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > > &&
             requires ( typename ::std::remove_reference_t< Tensor >::value_type v, Scalar s ) { v / s; } )
#else
template < class Tensor, class Scalar, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_is_scalar_divisible_v< ::std::remove_reference_t< Tensor >, ::std::remove_reference_t< Scalar > > > >
#endif
class scalar_division_tensor_expression;

// Scalar Modulo
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Scalar >
  requires ( LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > > &&
             requires ( typename ::std::remove_reference_t< Tensor >::value_type v, Scalar s ) { v % s; } )
#else
template < class Tensor, class Scalar, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_is_scalar_modulo_v< ::std::remove_reference_t< Tensor >, ::std::remove_reference_t< Scalar > > > >
#endif
class scalar_modulo_tensor_expression;

// Matrix Product
#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstMatrix, class SecondMatrix >
  requires ( LINALG_CONCEPTS::matrix_expression< ::std::remove_reference_t< FirstMatrix > > &&
             LINALG_CONCEPTS::matrix_expression< ::std::remove_reference_t< SecondMatrix > > &&
            ( requires ( const typename ::std::remove_reference_t< FirstMatrix >::value_type& v1, const typename ::std::remove_reference_t< SecondMatrix >::value_type& v2 ) { v1 * v2; } ) &&
             ( ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent(1) == ::std::remove_reference_t< SecondMatrix >::extents_type::static_extent(0) ) ||
               ( ::std::remove_reference_t< FirstMatrix >::extents_type::static_extent(1) == ::std::dynamic_extent ) ||
               ( ::std::remove_reference_t< SecondMatrix >::extents_type::static_extent(0) == ::std::dynamic_extent ) ) )
#else
template < class FirstMatrix, class SecondMatrix, typename = ::std::enable_if_t< LINALG_CONCEPTS::matrix_expression_v< ::std::remove_reference_t< FirstMatrix > > &&
                                                                                 LINALG_CONCEPTS::matrix_expression_v< ::std::remove_reference_t< SecondMatrix > > &&
                                                                                 LINALG_CONCEPTS::elements_are_multiplicative_v< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > > &&
                                                                                 LINALG_CONCEPTS::matrices_may_be_multiplicative_v< ::std::remove_reference_t< FirstMatrix >, ::std::remove_reference_t< SecondMatrix > > > >
#endif
class matrix_product_expression;

// Vector Matrix Product
#ifdef LINALG_ENABLE_CONCEPTS
template < class Vector, class Matrix >
  requires ( LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< Vector > > &&
             LINALG_CONCEPTS::matrix_expression< ::std::remove_reference_t< Matrix > > &&
             ( requires ( const typename ::std::remove_reference_t< Vector >::value_type& v1, const typename ::std::remove_reference_t< Matrix >::value_type& v2 ) { v1 * v2; } ) &&
             ( ( ::std::remove_reference_t< Vector >::extents_type::static_extent(0) == ::std::remove_reference_t< Matrix >::extents_type::static_extent(0) ) ||
               ( ::std::remove_reference_t< Vector >::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
               ( ::std::remove_reference_t< Matrix >::extents_type::static_extent(0) == ::std::dynamic_extent ) ) )
#else
template < class Vector, class Matrix, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< Vector > > &&
                                                                      LINALG_CONCEPTS::matrix_expression_v< ::std::remove_reference_t< Matrix > > &&
                                                                      LINALG_CONCEPTS::elements_are_multiplicative_v< ::std::remove_reference_t< Vector >, ::std::remove_reference_t< Matrix > > &&
                                                                      LINALG_CONCEPTS::vector_matrix_may_be_multiplicative_v< ::std::remove_reference_t< Vector >, ::std::remove_reference_t< Matrix > > > >
#endif
class vector_matrix_product_expression;

// Matrix Vector Product
#ifdef LINALG_ENABLE_CONCEPTS
template < class Matrix, class Vector >
  requires ( LINALG_CONCEPTS::matrix_expression< ::std::remove_reference_t< Matrix > > &&
             LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< Vector > > &&
             ( requires ( const typename ::std::remove_reference_t< Matrix >::value_type& v1, const typename ::std::remove_reference_t< Vector >::value_type& v2 ) { v1 * v2; } ) &&
             ( ( ::std::remove_reference_t< Vector >::extents_type::static_extent(0) == ::std::remove_reference_t< Matrix >::extents_type::static_extent(1) ) ||
               ( ::std::remove_reference_t< Vector >::extents_type::static_extent(0) == ::std::dynamic_extent ) ||
               ( ::std::remove_reference_t< Matrix >::extents_type::static_extent(1) == ::std::dynamic_extent ) ) )
#else
template < class Matrix, class Vector, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< Vector > > &&
                                                                      LINALG_CONCEPTS::matrix_expression_v< ::std::remove_reference_t< Matrix > > &&
                                                                      LINALG_CONCEPTS::elements_are_multiplicative_v< ::std::remove_reference_t< Vector >, ::std::remove_reference_t< Matrix > > &&
                                                                      LINALG_CONCEPTS::matrix_vector_may_be_multiplicative_v< ::std::remove_reference_t< Matrix >, ::std::remove_reference_t< Vector > > > >
#endif
class matrix_vector_product_expression;

// Outer Product
#ifdef LINALG_ENABLE_CONCEPTS
template < class FirstVector, class SecondVector >
  requires ( LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< FirstVector > > &&
             LINALG_CONCEPTS::vector_expression< ::std::remove_reference_t< SecondVector > > &&
            ( requires ( const typename ::std::remove_reference_t< FirstVector >::value_type& v1, const typename ::std::remove_reference_t< SecondVector >::value_type& v2 ) { v1 * v2; } ) )
#else
template < class FirstVector, class SecondVector, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< FirstVector > > &&
                                                                                 LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< SecondVector > > &&
                                                                                 LINALG_CONCEPTS::elements_are_multiplicative_v< ::std::remove_reference_t< FirstVector >, ::std::remove_reference_t< SecondVector > > > >
#endif
class outer_product_expression;

LINALG_EXPRESSIONS_END // end expressions namespace

LINALG_BEGIN // linalg namespace

// Fixed Size Tensor
template < class T,
           class Extents,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::default_accessor< T > >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( Extents::rank_dynamic() == 0 )
#endif
class fs_tensor;

// Dynamic Resizable Tensor
template < class T,
           class Extents,
           class LayoutPolicy   = default_layout,
           class CapExtents     = Extents,
           class Allocator      = ::std::allocator< T >,
           class AccessorPolicy = ::std::default_accessor< T > >
class dr_tensor;

// Dynamic Tensor
template < class         T,
           ::std::size_t N,
           class LayoutPolicy   = default_layout,
           class Allocator      = ::std::allocator< T >,
           class AccessorPolicy = ::std::default_accessor< T > >
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
           class AccessorPolicy = ::std::default_accessor< ElementType > >
using tensor_view = ::std::experimental::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >;

// Alias matrix view
template < class ElementType,
           auto  R,
           auto  C,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::default_accessor< ElementType > >
using matrix_view = tensor_view< ElementType, ::std::extents< ::std::common_type_t< decltype(R), decltype(C) >,static_cast< ::std::size_t >(R), static_cast< ::std::size_t >(C)>, LayoutPolicy, AccessorPolicy >;

// Alias matrix view
template < class ElementType,
           auto  N,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::default_accessor<ElementType> >
using vector_view = tensor_view< ElementType, ::std::extents< decltype(N) ,static_cast< ::std::size_t >(N) >, LayoutPolicy, AccessorPolicy >;

// Alias for dr_matrix
template < class T,
           auto  R,
           auto  C,
           class LayoutPolicy   = default_layout,
           auto  Rc             = R,
           auto  Cc             = C,
           class Allocator      = ::std::allocator< T >,
           class AccessorPolicy = ::std::default_accessor< T > >
using dr_matrix = dr_tensor< T, ::std::extents< ::std::common_type_t< decltype(R), decltype(C) >, static_cast< ::std::size_t >(R), static_cast< ::std::size_t >(C) >, LayoutPolicy, ::std::extents< ::std::common_type_t< decltype(Rc), decltype(Cc) >, static_cast<::std::size_t>(Rc), static_cast<::std::size_t>(Cc) >, Allocator, AccessorPolicy >;
template < class T >
using dyn_matrix = dr_matrix< T, ::std::dynamic_extent, ::std::dynamic_extent >;

// Alias for dr_vector
template < class T,
           auto  N,
           class LayoutPolicy   = default_layout,
           auto  Nc             = N,
           class Allocator      = ::std::allocator< T >,
           class AccessorPolicy = ::std::default_accessor< T > >
using dr_vector = dr_tensor< T, ::std::extents< decltype(N), static_cast< ::std::size_t >(N) >, LayoutPolicy, ::std::extents< decltype(Nc) ,static_cast< ::std::size_t >(Nc) >, Allocator, AccessorPolicy >;
template < class T >
using dyn_vector = dr_vector< T, ::std::dynamic_extent >;

// Alias for fs_matrix
template < class T,
           auto  R,
           auto  C,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::default_accessor< T > >
using fs_matrix = fs_tensor< T, ::std::extents< ::std::common_type_t< decltype(R), decltype(C) >, static_cast< ::std::size_t >(R), static_cast< ::std::size_t >(C) >, LayoutPolicy, AccessorPolicy >;

// Alias for fs_vector
template < class T,
           auto  N,
           class LayoutPolicy   = default_layout,
           class AccessorPolicy = ::std::default_accessor< T > >
using fs_vector = fs_tensor< T, ::std::extents< decltype(N), static_cast< ::std::size_t >(N) >, LayoutPolicy, AccessorPolicy >;

// Evaluates expressions
template < class T >
[[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION decltype(auto) eval( T&& t ) noexcept { return t; }

#ifdef LINALG_ENABLE_CONCEPTS
template < class T >  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< T > >
#else
template < class T, typename = ::std::enable_if_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< T > > > >
#endif
[[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION auto eval( T&& t ) noexcept( noexcept( ::std::declval<T&&>().evaluate() ) ) { return t.evaluate(); }

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP
