#ifndef LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
#define LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{
namespace concepts
{

#ifdef LINALG_ENABLE_CONCEPTS

//=================================================================================================
//  Tensor Concepts
//=================================================================================================

// The tensor concept.
template < class T >
concept tensor_expression = requires
{
  // Types
  typename T::value_type;
  typename T::index_type;
  typename T::size_type;
  typename T::extents_type;
  typename T::rank_type;
} &&
( LINALG_DETAIL::is_extents_v<typename T::extents_type> ) &&
requires( const T& t, typename T::rank_type n ) // Functions
{
  // Size functions
  { t.extent( n ) } noexcept -> ::std::same_as<typename T::size_type>;
  { T::rank() }     noexcept -> ::std::same_as<typename T::rank_type>;
  // Member accessors
  { t.extents() }   noexcept -> ::std::convertible_to<typename T::extents_type>;
  // Constexpr functions
  ::std::integral_constant< typename T::rank_type, T::rank() >::value;
} &&
requires( T& t, auto ... indices ) /* NOTE: there might be a way to enforce index_type ... instead of auto ... using C++17 style enable_if */
{
  // Index access
  #if LINALG_USE_BRACKET_OPERATOR
  { t.operator[]( indices ... ) } -> ::std::convertible_to<typename T::value_type>;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { t.operator()( indices ... ) } -> ::std::convertible_to<typename T::value_type>;
  #endif
};

// Matrix expression concept
template < class T >
concept matrix_expression =
tensor_expression< T > &&
( T::rank() == 2 );

// Vector expression concept
template < class T >
concept vector_expression =
tensor_expression< T > &&
( T::rank() == 1 );

// Readable tensor concept
template < class T >
concept readable_tensor =
tensor_expression<T> &&
requires
{
  // Types
  typename T::layout_type;
  typename T::mapping_type;
  typename T::accessor_type;
  typename T::reference;
  typename T::data_handle_type;
} &&
requires( const T& t, typename T::rank_type n ) // Functions
{
  // Size functions
  { T::rank_dynamic() }         noexcept -> ::std::same_as<typename T::rank_type>;
  { T::static_extent( n ) }     noexcept -> ::std::same_as<typename T::size_type>;
  { t.size() }                  noexcept -> ::std::same_as<typename T::size_type>;
  // Layout functions
  { T::is_always_strided() }    noexcept -> ::std::same_as<bool>;
  { T::is_always_exhaustive() } noexcept -> ::std::same_as<bool>;
  { T::is_always_unique() }     noexcept -> ::std::same_as<bool>;
  { t.is_strided() }            noexcept -> ::std::same_as<bool>;
  { t.is_exhaustive() }         noexcept -> ::std::same_as<bool>;
  { t.is_unique() }             noexcept -> ::std::same_as<bool>;
  { t.stride( n ) }             noexcept -> ::std::same_as<typename T::index_type>;
  // Member accessors
  { t.accessor() }              noexcept -> ::std::convertible_to<typename T::accessor_type>;
  { t.data_handle() }           noexcept -> ::std::convertible_to<typename T::data_handle_type>;
  { t.mapping() }               noexcept -> ::std::convertible_to<typename T::mapping_type>;
  // Constexpr functions
  integral_constant< typename T::rank_type, T::rank_dynamic() >::value;
  integral_constant< typename T::size_type, T::static_extent( n ) >::value;
  bool_constant< T::is_always_strided() >::value;
  bool_constant< T::is_always_exhaustive() >::value;
  bool_constant< T::is_always_unique() >::value;
};

// Writable tensor concept
template < class T >
concept writable_tensor =
readable_tensor<T> &&
requires( T& t, auto ... indices ) /* NOTE: there might be a way to enforce index_type ... instead of auto ... using C++17 style enable_if */
{
  // Index access
  #if LINALG_USE_BRACKET_OPERATOR
  { t.operator[]( indices ... ) } -> ::std::same_as<typename T::reference_type>;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { t.operator()( indices ... ) } -> ::std::same_as<typename T::reference_type>;
  #endif
};

// Static tensor concept
template < class T >
concept static_tensor =
writable_tensor<T> &&
( T::rank_dynamic() == 0 ) &&
requires ( const T& t )
{
  // Default constructor
  { T {} };
  // Constexpr functions
  LINALG_DETAIL::is_constexpr( []{ [[maybe_unused]] T { }; } ) &&
  integral_constant< typename T::size_type, t.size() >::value();
};

// Dynamic tensor concept
template < class T >
concept dynamic_tensor =
writable_tensor<T> &&
requires
{
  // Types
  typename T::allocator_type
} &&
requires( const T& t, typename T::extents_type s, typename T::allocator_type alloc ) // Functions
{
  // Size / Capacity
  { t.max_size() } -> ::std::same_as<typename T::size_type>;
  { t.capacity() } -> ::std::same_as<typename T::size_type>;
  { t.resize( s ) };
  // TBD on capacity_extents()
  // TBD on reserve( ... )
  // Allocator access
  { t.get_allocator() } -> ::std::same_as<typename T::allocator_type> ;
  // Constructors
  { T( alloc ) };
  { T( s, alloc ) };
};

// Unevaluated tensor expression
template < class T >
concept unevaluated_tensor_expression =
tensor_expression< T > &&
requires( T t )
{
  { t.operator auto() };
} &&
tensor_expression< decltype( auto( ::std::declval<T>() ) ) > &&
( static_tensor< decltype( ::std::declval<T>().operator auto() ) > ||
  dynamic_tensor< decltype( ::std::declval<T>().operator auto() ) > );

// Unary tensor expression concept
template < class T >
concept unary_tensor_expression =
unevaluated_tensor_expression< T > &&
requires ( T t )
{
  { t.underlying() } -> tensor_expression;
};

// Binary tensor expression concept

template < class T >
concept binary_tensor_expression =
unevaluated_tensor_expression< T > &&
requires ( T t )
{
  { t.first() } -> tensor_expression;
  { t.second() } -> tensor_expression;
};


#else

//- Tests for aliases

// Test if T has alias value_type
template < class T, class = void > struct has_value_type : public ::std::false_type { };
template < class T > struct has_value_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::value_type, typename T::value_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_value_type_v = has_value_type<T>::value;

// Test if T has alias index_type
template < class T, class = void > struct has_index_type : public ::std::false_type { };
template < class T > struct has_index_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::index_type, typename T::index_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_index_type_v = has_index_type<T>::value;

// Test if T has alias size_type
template < class T, class = void > struct has_size_type : public ::std::false_type { };
template < class T > struct has_size_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::size_type, typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_size_type_v = has_size_type<T>::value;

// Test if T has alias rank_type
template < class T, class = void > struct has_rank_type : public ::std::false_type { };
template < class T > struct has_rank_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::rank_type, typename T::rank_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_rank_type_v = has_rank_type<T>::value;

// Test if T has alias extents_type
template < class T, class = void > struct has_extents_type : public ::std::false_type { };
template < class T > struct has_extents_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::extents_type, typename T::extents_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_extents_type_v = has_extents_type<T>::value;

// Test if T has alias layout_type
template < class T, class = void > struct has_layout_type : public ::std::false_type { };
template < class T > struct has_layout_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::layout_type, typename T::layout_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_layout_type_v = has_layout_type<T>::value;

// Test if T has alias mapping_type
template < class T, class = void > struct has_mapping_type : public ::std::false_type { };
template < class T > struct has_mapping_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::mapping_type, typename T::mapping_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_mapping_type_v = has_mapping_type<T>::value;

// Test if T has alias accessor_type
template < class T, class = void > struct has_accessor_type : public ::std::false_type { };
template < class T > struct has_accessor_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::accessor_type, typename T::accessor_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_accessor_type_v = has_accessor_type<T>::value;

// Test if T has alias reference
template < class T, class = void > struct has_reference : public ::std::false_type { };
template < class T > struct has_reference< T, ::std::enable_if_t< ::std::is_same_v< typename T::reference, typename T::reference > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_reference_v = has_reference<T>::value;

// Test if T has alias data_handle_type
template < class T, class = void > struct has_data_handle_type : public ::std::false_type { };
template < class T > struct has_data_handle_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::data_handle_type, typename T::data_handle_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_data_handle_type_v = has_data_handle_type<T>::value;

// Test if T has alias allocator_type
template < class T, class = void > struct has_allocator_type : public ::std::false_type { };
template < class T > struct has_allocator_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::allocator_type, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_allocator_type_v = has_allocator_type<T>::value;

//- Test for functions

// Test for extent function
template < class T, class = void > struct has_extent_func : public ::std::false_type { };
template < class T > struct has_extent_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().extent( ::std::declval<typename T::rank_type>() ) ), typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_extent_func_v = has_extents_func<T>::value;

// Test for extents function
template < class T, class = void > struct has_extents_func : public ::std::false_type { };
template < class T > struct has_extents_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().extents() ), typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_extents_func_v = has_extents_func<T>::value;

// Test for rank function
template < class T, class = void > struct has_rank_func : public ::std::false_type { };
template < class T > struct has_rank_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::rank() ), typename T::rank_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_rank_func_v = has_rank_func<T>::value;

// Test for rank_dynamic function
template < class T, class = void > struct has_rank_dynamic_func : public ::std::false_type { };
template < class T > struct has_rank_dynamic_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::rank_dynamic() ), typename T::rank_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_rank_dynamic_func_v = has_rank_dynamic_func<T>::value;

// Test for static_extents function
template < class T, class = void > struct has_static_extents_func : public ::std::false_type { };
template < class T > struct has_static_extents_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::static_extents( ::std::declval<typename T::rank_type>() ) ), typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_static_extents_func_v = has_static_extents_func<T>::value;

// Test for size function
template < class T, class = void > struct has_size_func : public ::std::false_type { };
template < class T > struct has_size_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().size() ), typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_size_func_v = has_size_func<T>::value;

// Test for is_always_strided function
template < class T, class = void > struct has_is_always_strided_func : public ::std::false_type { };
template < class T > struct has_is_always_strided_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_always_strided() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_always_strided_func_v = has_is_always_strided_func<T>::value;

// Test for is_always_exhaustive function
template < class T, class = void > struct has_is_always_exhaustive_func : public ::std::false_type { };
template < class T > struct has_is_always_exhaustive_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_always_exhaustive() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_always_exhaustive_func_v = has_is_always_exhaustive_func<T>::value;

// Test for is_always_unique function
template < class T, class = void > struct has_is_always_unique_func : public ::std::false_type { };
template < class T > struct has_is_always_unique_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_always_unique() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_always_unique_func_v = has_is_always_unique_func<T>::value;

// Test for is_strided function
template < class T, class = void > struct has_is_strided_func : public ::std::false_type { };
template < class T > struct has_is_strided_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<T>().is_strided() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_strided_func_v = has_is_strided_func<T>::value;

// Test for is_exhaustive function
template < class T, class = void > struct has_is_exhaustive_func : public ::std::false_type { };
template < class T > struct has_is_exhaustive_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_exhaustive() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_exhaustive_func_v = has_is_exhaustive_func<T>::value;

// Test for is_unique function
template < class T, class = void > struct has_is_unique_func : public ::std::false_type { };
template < class T > struct has_is_unique_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_unique() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_unique_func_v = has_is_unique_func<T>::value;

// Test for stride function
template < class T, class = void > struct has_stride_func : public ::std::false_type { };
template < class T > struct has_stride_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::stride( ::std::declval<typename T::rank_type>() ) ), typename T::index_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_stride_func_v = has_stride_func<T>::value;

// Test for accessor function
template < class T, class = void > struct has_accessor_func : public ::std::false_type { };
template < class T > struct has_accessor_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::remove_cv_t< decltype( T::accessor() ) >, typename T::accessor_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_accessor_func_v = has_accessor_func<T>::value;

// Test for data_handle function
template < class T, class = void > struct has_data_handle_func : public ::std::false_type { };
template < class T > struct has_data_handle_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::remove_cv_t< decltype( T::data_handle() ) >, typename T::data_handle_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_data_handle_func_v = has_data_handle_func<T>::value;

// Test for mapping function
template < class T, class = void > struct has_mapping_func : public ::std::false_type { };
template < class T > struct has_mapping_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::remove_cv_t< decltype( T::mapping() ) >, typename T::mapping_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_mapping_func_v = has_mapping_func<T>::value;

// Test for max_size function
template < class T, class = void > struct has_max_size_func : public ::std::false_type { };
template < class T > struct has_max_size_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().max_size() ), typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_max_size_func_v = has_max_size_func<T>::value;

// Test for capacity function
template < class T, class = void > struct has_capacity_func : public ::std::false_type { };
template < class T > struct has_capacity_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().capacity() ), typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_capacity_func_v = has_capacity_func<T>::value;

// Test for resize function
template < class T, class = void > struct has_resize_func : public ::std::false_type { };
template < class T > struct has_resize_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().resize( ::std::declval<typename T::size_type>() ) ), void > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_resize_func_v = has_resize_func<T>::value;

// Test for get_allocator function
template < class T, class = void > struct has_get_allocator_func : public ::std::false_type { };
template < class T > struct has_get_allocator_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().get_allocator() ), const typename T::allocator_type& > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_get_allocator_func_v = has_get_allocator_func<T>::value;

// Test for construct from allocator
template < class T, class = void > struct constructible_from_alloc : public ::std::false_type { };
template < class T > struct constructible_from_alloc< T, ::std::enable_if_t< ::std::is_constructible_v< T, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool constructible_from_alloc_v = constructible_from_alloc<T>::value;

// Test for construct from size and allocator
template < class T, class = void > struct constructible_from_size_and_alloc : public ::std::false_type { };
template < class T > struct constructible_from_size_and_alloc< T, ::std::enable_if_t< ::std::is_constructible_v< T, typename T::extents_type, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool constructible_from_size_and_alloc_v = constructible_from_size_and_alloc<T>::value;

// Test for resize function
template < class T, class = void > struct has_resize_func : public ::std::false_type { };
template < class T > struct has_resize_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().resize( ::std::declval<typename T::size_type>() ) ), void > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_resize_func_v = has_resize_func<T>::value;

// Test for convertible bracket operator
template < class T, class = void > struct has_convertible_bracket_operator : public ::std::false_type { };
template < class T > struct has_convertible_bracket_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval<const T>().operator[]( ::std::declval<auto ...>() ) ), typename T::value_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_convertible_bracket_operator_v = has_convertible_bracket_operator<T>::value;

// Test for convertible paren operator
template < class T, class = void > struct has_convertible_paren_operator : public ::std::false_type { };
template < class T > struct has_convertible_paren_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval<const T>().operator()( ::std::declval<auto ...>() ) ), typename T::value_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_convertible_paren_operator_v = has_convertible_paren_operator<T>::value;

// Test for bracket operator
template < class T, class = void > struct has_bracket_operator : public ::std::false_type { };
template < class T > struct has_bracket_operator< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().operator[]( ::std::declval<auto ...>() ) ), typename T::reference_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_bracket_operator_v = has_bracket_operator<T>::value;

// Test for paren operator
template < class T, class = void > struct has_paren_operator : public ::std::false_type { };
template < class T > struct has_paren_operator< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().operator()( ::std::declval<auto ...>() ) ), typename T::referenec_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_paren_operator_v = has_paren_operator<T>::value;

// Test for assignable bracket operator
template < class T, class = void > struct has_assignable_bracket_operator : public ::std::false_type { };
template < class T > struct has_assignable_bracket_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval<const T>().operator[]( ::std::declval<auto ...>() ) = ::std::declval<typename T::value_type>() ), typename T::value_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_assignable_bracket_operator_v = has_assignable_bracket_operator<T>::value;

// Test for assignable paren operator
template < class T, class = void > struct has_assignable_paren_operator : public ::std::false_type { };
template < class T > struct has_assignable_paren_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval<const T>().operator()( ::std::declval<auto ...>() ) = ::std::declval<typename T::value_type>() ), typename T::value_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_assignable_paren_operator_v = has_assignable_paren_operator<T>::value;

// Test for underlying function
template < class T, class = void > struct has_underlying_func : public ::std::false_type { };
template < class T > struct has_underlying_func< T, ::std::enable_if_t< tensor_expression_v< decltype( ::std::declval<T>().underlying() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_underlying_func_v = has_underlying_func<T>::value;

// Test for first function
template < class T, class = void > struct has_first_func : public ::std::false_type { };
template < class T > struct has_first_func< T, ::std::enable_if_t< tensor_expression_v< decltype( ::std::declval<T>().first() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_first_func_v = has_first_func<T>::value;

// Test for second function
template < class T, class = void > struct has_second_func : public ::std::false_type { };
template < class T > struct has_second_func< T, ::std::enable_if_t< tensor_expression_v< decltype( ::std::declval<T>().second() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_second_func_v = has_second_func<T>::value;

//- Test for tensors

// Tensor expression
template < class T > struct tensor_expression : public ::std::conditional_t< 
  has_value_type_v<T> &&
  has_index_type_v<T> &&
  has_size_type_v<T> &&
  has_extents_type_v<T> &&
  has_rank_type_v<T> &&
#if LINALG_USE_BRACKET_OPERATOR
  has_convertible_bracket_operator<T> &&
#endif
#if LINALG_USE_PAREN_OPERATOR
  has_convertible_paren_operator<T> &&
#endif
  has_rank_func_v<T> &&
#if LINALG_HAS_CXX_20
  LINALG_DETAIL::is_constexpr( []{ [[maybe_unused]] decltype( T::rank() ) nodiscard_warning = T::rank(); } ) &&
#endif
  has_extents_func_v<T> &&
  has_extent_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool tensor_expression_v = tensor_expression<T>::value;

// Matrix expression
template < class T > struct matrix_expression : public ::std::conditional_t<
  tensor_expression_v<T> &&
  ( T::rank() == 2 ), ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool matrix_expression_v = matrix_expression<T>::value;

// Vector expression
template < class T > struct vector_expression : public ::std::conditional_t<
  tensor_expression_v<T> &&
  ( T::rank() == 1 ), ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool vector_expression_v = vector_expression<T>::value;

// Readable tensor
template < class T > struct readable_tensor : public ::std::condition_t<
  tensor_expression_v<T> &&
  has_layout_type_v<T> &&
  has_mapping_type_v<T> &&
  has_accessor_type_v<T> &&
  has_reference_v<T>
  has_data_handle_type_v<T> &&
  has_rank_dynamic_func_v<T> &&
  has_static_extents_func_v<T> &&
  has_size_func_v<T> &&
  has_is_always_strided_func_v<T> &&
  has_is_always_exhaustive_func_v<T> &&
  has_is_always_unique_func_v<T> &&
  has_is_strided_func_v<T> &&
  has_is_exhaustive_func_v<T> &&
  has_is_unique_func_v<T> &&
#if LINALG_HAS_CXX_20
  LINALG_DETAIL::is_constexpr( []{ [[maybe_unused]] decltype( T::rank_dynamic() ) nodiscard_warning = T::rank_dynamic(); } ) &&
  LINALG_DETAIL::is_constexpr( []{ [[maybe_unused]] decltype( T::static_extent( T::rank() ) ) nodiscard_warning = T::static_extent( T::rank() ); } ) &&
  LINALG_DETAIL::is_constexpr( []{ [[maybe_unused]] bool nodiscard_warning = T::is_always_strided(); } ) &&
  LINALG_DETAIL::is_constexpr( []{ [[maybe_unused]] bool nodiscard_warning = T::is_always_exhaustive(); } ) &&
  LINALG_DETAIL::is_constexpr( []{ [[maybe_unused]] bool nodiscard_warning = T::is_always_unique(); } ) &&
#endif
  has_stride_func_v<T> &&
  has_accessor_func_v<T> &&
  has_data_handle_func_v<T> &&
  has_mapping_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool readable_tensor_v = readable_tensor<T>::value;

// Writable tensor
template < class T > struct writable_tensor : public ::std::conditional_t<
  readable_tensor_v<T>
#if LINALG_USE_BRACKET_OPERATOR
  && has_bracket_operator<T>
#endif
#if LINALG_USE_PAREN_OPERATOR
  && has_paren_operator<T>
#endif
  , ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool writable_tensor_v = writable_tensor<T>::value;

// Static tensor
template < class T > struct static_tensor : public ::std::conditional_t<
  writable_tensor_v<T> &&
  ( T::rank_dynamic() == 0 ) &&
  ::std::is_default_constructible_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool static_tensor_v = static_tensor<T>::value;

// Dynamic tensor
template < class T > struct dynamic_tensor : public ::std::conditional_t<
  writable_tensor_v<T> &&
  has_allocator_type_v<T> &&
  has_max_size_func_v<T> &&
  has_capacity_func_v<T> &&
  has_resize_func_v<T> &&
  has_get_allocator_v<T> &&
  constructible_from_alloc_v<T> &&
  constructible_from_size_and_alloc_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool dynamic_tensor_v = dynamic_tensor<T>::value;

// Unary tensor expression
template < class T > struct unary_tensor_expression : public ::std::conditional_t<
  tensor_expression_v<T> &&
  has_underlying_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool unary_tensor_expression_v = unary_tensor_expression<T>::value;

// Binary tensor expression
template < class T > struct unary_tensor_expression : public ::std::conditional_t<
  tensor_expression_v<T> &&
  has_first_func_v<T> &&
  has_second_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool binary_tensor_expression_v = binary_tensor_expression<T>::value;

#endif

}       //- concepts namespace
}       //- experimental namespace
}       //- std namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
