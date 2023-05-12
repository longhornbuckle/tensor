//==================================================================================================
//  File:       forward_declarations.hpp
//
//  Summary:    This header forward declares the primary linear algebra classes.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP
#define LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{

// Default layout
using default_layout = ::std::experimental::layout_right;

// Unary Tensor Expressions

// Negate
template < tensor_expression Tensor >
class negate_tensor_expression;

// Transpose
template < tensor_expression Tensor >
class transpose_tensor_expression;

// Conjugate
template < tensor_expression Tensor >
class conjugate_tensor_expression;

// Binary Tensor Expression

// Addition
template < tensor_expression FirstTensor, tensor_expression SecondTensor >
class add_tensor_expression
  requires ( ( FirstTensor::rank() == SecondTensor::rank() ) &&
             LINALG_DETAIL::extents_maybe_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > );

// Subtraction
template < tensor_expression FirstTensor, tensor_expression SecondTensor >
class subtraction_tensor_expression
  requires ( ( FirstTensor::rank() == SecondTensor::rank() ) &&
             LINALG_DETAIL::extents_maybe_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > );

// Scalar Pre-Multiply
template < class ValueType, tensor_expression Tensor >
class scalar_preprod_tensor_expression
  requires requires ( const ValueType& v1, const typename Tensor::value_type& v2 ) { { v1 * v2; } };



// Tensor
template < class T,
           class Extents,
           class LayoutPolicy   = default_layout,
           class CapExtents     = Extents,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
class tensor;

namespace math
{

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

// Alias tensor
template < class T,
           class Extents,
           class LayoutPolicy   = default_layout,
           class CapExtents     = Extents,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
using tensor = ::std::experimental::tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>;

// Alias for matrix
template < class T,
           auto  R,
           auto  C,
           class LayoutPolicy   = default_layout,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
using matrix = tensor< T, ::std::experimental::extents<::std::common_type_t<decltype(R),decltype(C)>,static_cast<::std::size_t>(R),static_cast<::std::size_t>(C)>, LayoutPolicy, ::std::experimental::extents<::std::common_type_t<decltype(R),decltype(C)>,static_cast<::std::size_t>(R),static_cast<::std::size_t>(C)>, Allocator, AccessorPolicy >;

// Alias for vector
template < class T,
           auto  N,
           class LayoutPolicy   = default_layout,
           class Allocator      = ::std::allocator<T>,
           class AccessorPolicy = ::std::experimental::default_accessor<T> >
using vector = tensor< T, ::std::experimental::extents<decltype(N),static_cast<::std::size_t>(N)>, LayoutPolicy, ::std::experimental::extents<decltype(N),static_cast<::std::size_t>(N)>, Allocator, AccessorPolicy >;

// namespace detail
// {
// [[nodiscard]] constexpr ::std::size_t dyn_ext( [[maybe_unused]] size_t i ) noexcept { return ::std::experimental::dynamic_extent; }

// template < class T,
//            class LayoutPolicy,
//            class Allocator,
//            class AccessorPolicy,
//            class Seq >
// struct dyn_tensor_impl;

// template < class T,
//            class LayoutPolicy,
//            class Allocator,
//            class AccessorPolicy,
//            size_t ... Indices >
// struct dyn_tensor_impl< T, LayoutPolicy, Allocator, AccessorPolicy, ::std::index_sequence<Indices...> >
// {
//   using type = tensor< T, ::std::experimental::extents<::std::size_t,dyn_ext(Indices) ...>, LayoutPolicy, Allocator, AccessorPolicy >;
// };

// }

// // Alias for dynamic tensor
// template < class T,
//            size_t Rank,
//            class LayoutPolicy   = default_layout,
//            class Allocator      = ::std::allocator<T>,
//            class AccessorPolicy = ::std::experimental::default_accessor<T> >
// using dyn_tensor = typename detail::dyn_tensor_impl< T, LayoutPolicy, Allocator, AccessorPolicy, ::std::make_index_sequence<Rank> >::type;

// // Alias for dynamic matrix
// template < class T,
//            class LayoutPolicy   = default_layout,
//            class Allocator      = ::std::allocator<T>,
//            class AccessorPolicy = ::std::experimental::default_accessor<T> >
// using dyn_matrix = matrix< T, ::std::experimental::dynamic_extent, ::std::experimental::dynamic_extent, LayoutPolicy, Allocator, AccessorPolicy >;

// // Alias for dynamic vector
// template < class T,
//            class LayoutPolicy   = default_layout,
//            class Allocator      = ::std::allocator<T>,
//            class AccessorPolicy = ::std::experimental::default_accessor<T> >
// using dyn_vector = vector< T, ::std::experimental::dynamic_extent, LayoutPolicy, Allocator, AccessorPolicy >;

// // Alias for fixed size tensor
// template < class T,
//            class Extents,
//            class LayoutPolicy   = default_layout,
//            class AccessorPolicy = ::std::experimental::default_accessor<T> >
//  using fixed_size_tensor = tensor< T, Extents, LayoutPolicy, util::fixed_size_allocator< T, LayoutPolicy::template mapping<Extents>( Extents() ).required_span_size() >, AccessorPolicy >;

//  // Alias for fixed size matrix
//  template < class T,
//             auto  R,
//             auto  C,
//             class LayoutPolicy   = default_layout,
//             class AccessorPolicy = ::std::experimental::default_accessor<T> >
// using fixed_size_matrix = matrix< T, R, C, LayoutPolicy, util::fixed_size_allocator< T, LayoutPolicy::template mapping< ::std::experimental::extents< ::std::size_t, R, C > >( ::std::experimental::extents< ::std::size_t, R, C >() ).required_span_size() >, AccessorPolicy >;

// // Alias for fixed size vector
//  template < class T,
//             auto  N,
//             class LayoutPolicy   = default_layout,
//             class AccessorPolicy = ::std::experimental::default_accessor<T> >
// using fixed_size_vector = vector< T, N, LayoutPolicy, util::fixed_size_allocator< T, LayoutPolicy::template mapping< ::std::experimental::extents< ::std::size_t, N > >( ::std::experimental::extents< ::std::size_t, N >() ).required_span_size() >, AccessorPolicy >;


}       //- math namespace
}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP
