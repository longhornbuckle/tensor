#ifndef LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
#define LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP

#include <experimental/linear_algebra.hpp>

LINALG_CONCEPTS_BEGIN // concepts namespace

#ifdef LINALG_ENABLE_CONCEPTS

//=================================================================================================
//  Tensor Concepts
//=================================================================================================

template < class T, class IndexType, IndexType ... indices >
concept has_readable_index_operator =
requires( const T& t )
{
  // Index access
  #if LINALG_USE_BRACKET_OPERATOR
  { t.operator[]( indices ... ) } -> ::std::convertible_to< typename T::value_type >;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { t.operator()( indices ... ) } -> ::std::convertible_to< typename T::value_type >;
  #endif
};

template < class T, class IntegerSequence >
struct has_readable_index_operator_helper : public ::std::false_type { };

template < class T, class IndexType, IndexType ... indices >
struct has_readable_index_operator_helper< T, ::std::integer_sequence< IndexType, indices ... > >
  : public ::std::conditional_t< has_readable_index_operator< T, IndexType, indices ... >, ::std::true_type, ::std::false_type > { };

template < class T, class IndexType, IndexType ... indices >
concept has_writable_index_operator =
requires( const T& t )
{
  // Index access
  #if LINALG_USE_BRACKET_OPERATOR
  { t.operator[]( indices ... ) } -> ::std::same_as< typename T::reference >;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { t.operator()( indices ... ) } -> ::std::same_as< typename T::reference >;
  #endif
};

template < class T, class IntegerSequence >
struct has_writable_index_operator_helper : public ::std::false_type { };

template < class T, class IndexType, IndexType ... indices >
struct has_writable_index_operator_helper< T, ::std::integer_sequence< IndexType, indices ... > >
  : public ::std::conditional_t< has_writable_index_operator< T, IndexType, indices ... >, ::std::true_type, ::std::false_type > { };

// The tensor concept.
template < class T >
concept tensor_expression = requires
{
  // Types
  typename T::value_type;
  typename T::size_type;
  typename T::extents_type;
  typename T::rank_type;
} &&
( LINALG_DETAIL::is_extents_v< typename T::extents_type > ) &&
requires( const T& t, typename T::rank_type n ) // Functions
{
  // Size functions
  // NOTE: P0009r18 lists return type of extent(n) as size_type. Implementation returns index_type (for both mdspan and extents).
  { t.extent( n ) }          -> ::std::same_as< typename T::index_type >; // ::std::same_as< typename T::size_type >;
  // NOTE: POOO9r18 lists return type of rank() as rank_type. Implementation returns size_t (for mdspan only).
  { T::rank() }     noexcept -> ::std::convertible_to< typename T::rank_type >; // ::std::same_as< typename T::rank_type >;
  // Member accessors
  { t.extents() }            -> ::std::convertible_to< typename T::extents_type >;
  // Constexpr functions
  ::std::integral_constant< typename T::rank_type, T::rank() >::value;
} &&
has_readable_index_operator_helper< T, ::std::make_integer_sequence< typename T::size_type, T::rank() > >::value;

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
requires
{
  // Types
  typename T::layout_type;
  typename T::mapping_type;
  typename T::accessor_type;
  typename T::reference;
  typename T::data_handle_type;
} &&
requires( ::std::remove_cv_t< T >& t, typename T::rank_type n ) // Functions
{
  // Size functions
  // NOTE: POOO9r18 lists return type of rank() as rank_type. Implementation returns size_t (for mdspan only).
  { T::rank_dynamic() }         noexcept -> ::std::convertible_to< typename T::rank_type >; // ::std::same_as< typename T::rank_type >;
  { T::static_extent( n ) }     noexcept -> ::std::same_as< ::std::size_t >;
  // NOTE: POOO9r18 lists return type of size() as size_type. Implementation returns size_t.
  { t.size() }                           -> ::std::same_as< ::std::size_t >; // ::std::same_as< typename T::size_type >;
  // Layout functions
  { T::is_always_strided() }    noexcept -> ::std::same_as< bool >;
  { T::is_always_exhaustive() } noexcept -> ::std::same_as< bool >;
  { T::is_always_unique() }     noexcept -> ::std::same_as< bool >;
  { t.is_strided() }                     -> ::std::same_as< bool >;
  { t.is_exhaustive() }                  -> ::std::same_as< bool >;
  { t.is_unique() }                      -> ::std::same_as< bool >;
  // NOTE: POOO9r18 lists return type of stride(n) as size_type. Implementation returns index_type (for mdspan and mapping types).
  { t.stride( n ) }                      -> ::std::same_as< typename T::index_type >; // ::std::same_as< typename T::size_type >;
  // Member accessors
  { t.accessor() }                       -> ::std::convertible_to< typename T::accessor_type >;
  { t.data_handle() }                    -> ::std::convertible_to< typename T::data_handle_type >;
  { t.mapping() }                        -> ::std::convertible_to< typename T::mapping_type >;
  // Constexpr functions
  ::std::integral_constant< typename T::rank_type, T::rank_dynamic() >::value;
  ::std::integral_constant< ::std::size_t, T::static_extent( T::rank_dynamic() ) >::value;
  ::std::bool_constant< T::is_always_strided() >::value;
  ::std::bool_constant< T::is_always_exhaustive() >::value;
  ::std::bool_constant< T::is_always_unique() >::value;
};

// Writable tensor concept
template < class T >
concept writable_tensor =
readable_tensor< T > &&
has_writable_index_operator_helper< T, ::std::make_integer_sequence< typename T::size_type, T::rank() > >::value;

// Static tensor concept
template < class T >
concept static_tensor =
writable_tensor< T > &&
( T::rank_dynamic() == 0 ) &&
::std::default_initializable< T > &&
// Constexpr functions
requires
{
  ::std::integral_constant< typename T::size_type, T { }.size() >::value;
};

// Dynamic tensor concept
template < class T >
concept dynamic_tensor =
writable_tensor< T > &&
requires
{
  // Types
  typename T::allocator_type;
  typename T::capacity_extents_type;
} &&
requires( const T& ct, T& t, typename T::extents_type s, typename T::allocator_type alloc ) // Functions
{
  // Size / Capacity
  { ct.max_size() } -> ::std::same_as< typename T::size_type >;
  { ct.capacity() } -> ::std::same_as< typename T::capacity_extents_type >;
  { t.resize( s ) };
  // Allocator access
  { ct.get_allocator() } -> ::std::same_as< typename T::allocator_type > ;
} &&
constructible_from< T, typename T::allocator_type > &&
constructible_from< T, const typename T::extents_type&, typename T::allocator_type >;

// Unevaluated tensor expression
template < class T >
concept unevaluated_tensor_expression =
tensor_expression< T > &&
requires
{
  // Types
  typename T::evaluated_type;
} &&
requires( T t )
{
  { t.evaluate() };
} &&
tensor_expression< decltype( ::std::declval< T >().evaluate() ) > &&
( static_tensor< decltype( ::std::declval< T >().evaluate() ) > ||
  dynamic_tensor< decltype( ::std::declval< T >().evaluate() ) > );

// Unary tensor expression concept
template < class T >
concept unary_tensor_expression =
unevaluated_tensor_expression< T > &&
requires ( const T& t )
{
  { t.underlying() };
};

// Binary tensor expression concept

template < class T >
concept binary_tensor_expression =
unevaluated_tensor_expression< T > &&
requires ( const T& t )
{
  { t.first() };
  { t.second() };
};


#else

//- Tests for aliases

// Test if T has alias value_type
template < class T, class = void > struct has_value_type : public ::std::false_type { };
template < class T > struct has_value_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::value_type, typename T::value_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_value_type_v = has_value_type< T >::value;

// Test if T has alias index_type
template < class T, class = void > struct has_index_type : public ::std::false_type { };
template < class T > struct has_index_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::index_type, typename T::index_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_index_type_v = has_index_type< T >::value;

// Test if T has alias size_type
template < class T, class = void > struct has_size_type : public ::std::false_type { };
template < class T > struct has_size_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::size_type, typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_size_type_v = has_size_type< T >::value;

// Test if T has alias rank_type
template < class T, class = void > struct has_rank_type : public ::std::false_type { };
template < class T > struct has_rank_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::rank_type, typename T::rank_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_rank_type_v = has_rank_type< T >::value;

// Test if T has alias extents_type
template < class T, class = void > struct has_extents_type : public ::std::false_type { };
template < class T > struct has_extents_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::extents_type, typename T::extents_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_extents_type_v = has_extents_type< T >::value;

// Test if T has alias capacity_extents_type
template < class T, class = void > struct has_capacity_extents_type : public ::std::false_type { };
template < class T > struct has_capacity_extents_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::capacity_extents_type, typename T::capacity_extents_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_capacity_extents_type_v = has_capacity_extents_type< T >::value;

// Test if T has alias layout_type
template < class T, class = void > struct has_layout_type : public ::std::false_type { };
template < class T > struct has_layout_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::layout_type, typename T::layout_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_layout_type_v = has_layout_type< T >::value;

// Test if T has alias mapping_type
template < class T, class = void > struct has_mapping_type : public ::std::false_type { };
template < class T > struct has_mapping_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::mapping_type, typename T::mapping_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_mapping_type_v = has_mapping_type< T >::value;

// Test if T has alias accessor_type
template < class T, class = void > struct has_accessor_type : public ::std::false_type { };
template < class T > struct has_accessor_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::accessor_type, typename T::accessor_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_accessor_type_v = has_accessor_type< T >::value;

// Test if T has alias reference
template < class T, class = void > struct has_reference : public ::std::false_type { };
template < class T > struct has_reference< T, ::std::enable_if_t< ::std::is_same_v< typename T::reference, typename T::reference > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_reference_v = has_reference< T >::value;

// Test if T has alias data_handle_type
template < class T, class = void > struct has_data_handle_type : public ::std::false_type { };
template < class T > struct has_data_handle_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::data_handle_type, typename T::data_handle_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_data_handle_type_v = has_data_handle_type< T >::value;

// Test if T has alias allocator_type
template < class T, class = void > struct has_allocator_type : public ::std::false_type { };
template < class T > struct has_allocator_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::allocator_type, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_allocator_type_v = has_allocator_type< T >::value;

//- Test for functions

// Test for extent function
template < class T, class = void > struct has_extent_func : public ::std::false_type { };
template < class T > struct has_extent_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().extent( ::std::declval< typename T::rank_type >() ) ), typename T::index_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_extent_func_v = has_extent_func< T >::value;

// Test for extents function
template < class T, class = void > struct has_extents_func : public ::std::false_type { };
template < class T > struct has_extents_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::decay_t< decltype( ::std::declval< const T >().extents() ) >, typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_extents_func_v = has_extents_func< T >::value;

// Test for rank function
template < class T, class = void > struct has_rank_func : public ::std::false_type { };
template < class T > struct has_rank_func< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( T::rank() ), typename T::rank_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_rank_func_v = has_rank_func< T >::value;

// Test for constexpr rank function
template < class T, class = void > struct has_constexpr_rank_func : public ::std::false_type { };
template < class T > struct has_constexpr_rank_func< T, ::std::enable_if_t< ::std::is_convertible_v< ::std::integral_constant< typename T::rank_type, T::rank() >, ::std::integral_constant< typename T::rank_type, T::rank() > > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_constexpr_rank_func_v = has_constexpr_rank_func< T >::value;

// Test for rank_dynamic function
template < class T, class = void > struct has_rank_dynamic_func : public ::std::false_type { };
template < class T > struct has_rank_dynamic_func< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( T::rank_dynamic() ), typename T::rank_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_rank_dynamic_func_v = has_rank_dynamic_func< T >::value;

// Test for constexpr rank_dynamic function
template < class T, class = void > struct has_constexpr_rank_dynamic_func : public ::std::false_type { };
template < class T > struct has_constexpr_rank_dynamic_func< T, ::std::enable_if_t< ::std::is_convertible_v< integral_constant< typename T::rank_type, T::rank_dynamic() >, integral_constant< typename T::rank_type, T::rank_dynamic() > > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_constexpr_rank_dynamic_func_v = has_constexpr_rank_dynamic_func< T >::value;

// Test for static_extent function
template < class T, class = void > struct has_static_extent_func : public ::std::false_type { };
template < class T > struct has_static_extent_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::static_extent( ::std::declval< typename T::rank_type >() ) ), ::std::size_t > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_static_extent_func_v = has_static_extent_func< T >::value;

// Test for constexpr static_extent function
template < class T, class = void > struct has_constexpr_static_extent_func : public ::std::false_type { };
template < class T > struct has_constexpr_static_extent_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::integral_constant< ::std::size_t, T::static_extent( typename T::rank_type() ) >, ::std::integral_constant< ::std::size_t, T::static_extent( typename T::rank_type() ) > > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_constexpr_static_extent_func_v = has_constexpr_static_extent_func< T >::value;

// Test for size function
template < class T, class = void > struct has_size_func : public ::std::false_type { };
template < class T > struct has_size_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().size() ), ::std::size_t > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_size_func_v = has_size_func< T >::value;

// Test for is_always_strided function
template < class T, class = void > struct has_is_always_strided_func : public ::std::false_type { };
template < class T > struct has_is_always_strided_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_always_strided() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_always_strided_func_v = has_is_always_strided_func< T >::value;

// Test for constexpr is_always_strided function
template < class T, class = void > struct has_constexpr_is_always_strided_func : public ::std::false_type { };
template < class T > struct has_constexpr_is_always_strided_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::bool_constant< T::is_always_strided() >, ::std::bool_constant< T::is_always_strided() > > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_constexpr_is_always_strided_func_v = has_constexpr_is_always_strided_func< T >::value;

// Test for is_always_exhaustive function
template < class T, class = void > struct has_is_always_exhaustive_func : public ::std::false_type { };
template < class T > struct has_is_always_exhaustive_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_always_exhaustive() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_always_exhaustive_func_v = has_is_always_exhaustive_func< T >::value;

// Test for constexpr is_always_exhaustive function
template < class T, class = void > struct has_constexpr_is_always_exhaustive_func : public ::std::false_type { };
template < class T > struct has_constexpr_is_always_exhaustive_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::bool_constant< T::is_always_exhaustive() >, ::std::bool_constant< T::is_always_exhaustive() > > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_constexpr_is_always_exhaustive_func_v = has_constexpr_is_always_exhaustive_func< T >::value;

// Test for is_always_unique function
template < class T, class = void > struct has_is_always_unique_func : public ::std::false_type { };
template < class T > struct has_is_always_unique_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( T::is_always_unique() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_always_unique_func_v = has_is_always_unique_func< T >::value;

// Test for constexpr is_always_unique function
template < class T, class = void > struct has_constexpr_is_always_unique_func : public ::std::false_type { };
template < class T > struct has_constexpr_is_always_unique_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::bool_constant< T::is_always_unique() >, ::std::bool_constant< T::is_always_unique() > > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_constexpr_is_always_unique_func_v = has_constexpr_is_always_unique_func< T >::value;

// Test for is_strided function
template < class T, class = void > struct has_is_strided_func : public ::std::false_type { };
template < class T > struct has_is_strided_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< T >().is_strided() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_strided_func_v = has_is_strided_func< T >::value;

// Test for is_exhaustive function
template < class T, class = void > struct has_is_exhaustive_func : public ::std::false_type { };
template < class T > struct has_is_exhaustive_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< T >().is_exhaustive() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_exhaustive_func_v = has_is_exhaustive_func< T >::value;

// Test for is_unique function
template < class T, class = void > struct has_is_unique_func : public ::std::false_type { };
template < class T > struct has_is_unique_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< T >().is_unique() ), bool > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_is_unique_func_v = has_is_unique_func< T >::value;

// Test for stride function
template < class T, class = void > struct has_stride_func : public ::std::false_type { };
template < class T > struct has_stride_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< T >().stride( ::std::declval< typename T::rank_type >() ) ), typename T::index_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_stride_func_v = has_stride_func< T >::value;

// Test for accessor function
template < class T, class = void > struct has_accessor_func : public ::std::false_type { };
template < class T > struct has_accessor_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::decay_t< decltype( ::std::declval< ::std::remove_cv_t< T > >().accessor() ) >, typename T::accessor_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_accessor_func_v = has_accessor_func< T >::value;

// Test for data_handle function
template < class T, class = void > struct has_data_handle_func : public ::std::false_type { };
template < class T > struct has_data_handle_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::decay_t< decltype( ::std::declval< ::std::remove_cv_t< T > >().data_handle() ) >, typename T::data_handle_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_data_handle_func_v = has_data_handle_func< T >::value;

// Test for mapping function
template < class T, class = void > struct has_mapping_func : public ::std::false_type { };
template < class T > struct has_mapping_func< T, ::std::enable_if_t< ::std::is_same_v< ::std::decay_t< decltype( ::std::declval< const T >().mapping() ) >, typename T::mapping_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_mapping_func_v = has_mapping_func< T >::value;

// Test for max_size function
template < class T, class = void > struct has_max_size_func : public ::std::false_type { };
template < class T > struct has_max_size_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().max_size() ), typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_max_size_func_v = has_max_size_func< T >::value;

// Test for capacity function
template < class T, class = void > struct has_capacity_func : public ::std::false_type { };
template < class T > struct has_capacity_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().capacity() ), typename T::capacity_extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_capacity_func_v = has_capacity_func< T >::value;

// Test for resize function
template < class T, class = void > struct has_resize_func : public ::std::false_type { };
template < class T > struct has_resize_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().resize( ::std::declval< typename T::size_type >() ) ), void > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_resize_func_v = has_resize_func< T >::value;

// Test for get_allocator function
template < class T, class = void > struct has_get_allocator_func : public ::std::false_type { };
template < class T > struct has_get_allocator_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().get_allocator() ), const typename T::allocator_type& > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_get_allocator_func_v = has_get_allocator_func< T >::value;

// Test for construct from allocator
template < class T, class = void > struct constructible_from_alloc : public ::std::false_type { };
template < class T > struct constructible_from_alloc< T, ::std::enable_if_t< ::std::is_constructible_v< T, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool constructible_from_alloc_v = constructible_from_alloc< T >::value;

// Test for construct from size and allocator
template < class T, class = void > struct constructible_from_size_and_alloc : public ::std::false_type { };
template < class T > struct constructible_from_size_and_alloc< T, ::std::enable_if_t< ::std::is_constructible_v< T, typename T::extents_type, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool constructible_from_size_and_alloc_v = constructible_from_size_and_alloc< T >::value;

// // Test for convertible bracket operator
// template < class T, class = void > struct has_convertible_bracket_operator : public ::std::false_type { };
// template < class T > struct has_convertible_bracket_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval< const T >().operator[]( ::std::declval< auto ... >() ) ), typename T::value_type > > > : public ::std::true_type { };
// template < class T > inline constexpr bool has_convertible_bracket_operator_v = has_convertible_bracket_operator< T >::value;

// // Test for convertible paren operator
// template < class T, class = void > struct has_convertible_paren_operator : public ::std::false_type { };
// template < class T > struct has_convertible_paren_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval< const T >().operator()( ::std::declval< auto ... >() ) ), typename T::value_type > > > : public ::std::true_type { };
// template < class T > inline constexpr bool has_convertible_paren_operator_v = has_convertible_paren_operator< T >::value;

// // Test for bracket operator
// template < class T, class = void > struct has_bracket_operator : public ::std::false_type { };
// template < class T > struct has_bracket_operator< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().operator[]( ::std::declval< auto ... >() ) ), typename T::reference_type > > > : public ::std::true_type { };
// template < class T > inline constexpr bool has_bracket_operator_v = has_bracket_operator< T >::value;

// // Test for paren operator
// template < class T, class = void > struct has_paren_operator : public ::std::false_type { };
// template < class T > struct has_paren_operator< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< const T >().operator()( ::std::declval< auto ... >() ) ), typename T::referenec_type > > > : public ::std::true_type { };
// template < class T > inline constexpr bool has_paren_operator_v = has_paren_operator< T >::value;

// // Test for assignable bracket operator
// template < class T, class = void > struct has_assignable_bracket_operator : public ::std::false_type { };
// template < class T > struct has_assignable_bracket_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval< const T >().operator[]( ::std::declval< auto ... >() ) = ::std::declval< typename T::value_type >() ), typename T::value_type > > > : public ::std::true_type { };
// template < class T > inline constexpr bool has_assignable_bracket_operator_v = has_assignable_bracket_operator< T >::value;

// // Test for assignable paren operator
// template < class T, class = void > struct has_assignable_paren_operator : public ::std::false_type { };
// template < class T > struct has_assignable_paren_operator< T, ::std::enable_if_t< ::std::is_convertible_v< decltype( ::std::declval< const T >().operator()( ::std::declval< auto ... >() ) = ::std::declval< typename T::value_type >() ), typename T::value_type > > > : public ::std::true_type { };
// template < class T > inline constexpr bool has_assignable_paren_operator_v = has_assignable_paren_operator< T >::value;

// Test for evaluate function
template < class T, class = void > struct has_evaluate_func : public ::std::false_type { };
template < class T > struct has_evaluate_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< ::std::remove_reference_t< const T > >().evaluate() ), decltype( ::std::declval< ::std::remove_reference_t< const T > >().evaluate() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_evaluate_func_v = has_evaluate_func< T >::value;

// Test for underlying function
template < class T, class = void > struct has_underlying_func : public ::std::false_type { };
template < class T > struct has_underlying_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< T >().underlying() ), decltype( ::std::declval< T >().underlying() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_underlying_func_v = has_underlying_func< T >::value;

// Test for first function
template < class T, class = void > struct has_first_func : public ::std::false_type { };
template < class T > struct has_first_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< T >().first() ), decltype( ::std::declval< T >().first() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_first_func_v = has_first_func< T >::value;

// Test for second function
template < class T, class = void > struct has_second_func : public ::std::false_type { };
template < class T > struct has_second_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< T >().second() ), decltype( ::std::declval< T >().second() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_second_func_v = has_second_func< T >::value;

// Test for equal extents
template < class T1, class T2, class = void > struct may_have_equal_extents : public ::std::false_type { };
template < class T1, class T2 > struct may_have_equal_extents< T1, T2, ::std::enable_if_t< LINALG_DETAIL::extents_may_be_equal_v< typename T1::extents_type, typename T2::extents_type > > > : public ::std::true_type { };
template < class T1, class T2 > inline constexpr bool may_have_equal_extents_v = may_have_equal_extents< T1, T2 >::value;

// Test for equal ranks
template < class T1, class T2, class = void > struct has_equal_ranks : public ::std::false_type { };
template < class T1, class T2 > struct has_equal_ranks< T1, T2, ::std::enable_if_t< ( T1::rank() == T2::rank() ) > > : public ::std::true_type { };
template < class T1, class T2 > inline constexpr bool has_equal_ranks_v = has_equal_ranks< T1, T2 >::value;

// Test for rank
template < class T, ::std::size_t R, class = void > struct has_rank : public ::std::false_type { };
template < class T, ::std::size_t R > struct has_rank< T, R, ::std::enable_if_t< T::rank() == R > > : public ::std::true_type { };
template < class T, ::std::size_t R > inline constexpr bool has_rank_v = has_rank< T, R >::value;

// Test for additive elements
template < class T1, class T2, class = void > struct elements_are_additive : public ::std::false_type { };
template < class T1, class T2 > struct elements_are_additive< T1, T2, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< typename T1::value_type >() + ::std::declval< typename T2::value_type >() ), decltype( ::std::declval< typename T1::value_type >() + ::std::declval< typename T2::value_type >() ) > > > : public ::std::true_type { };
template < class T1, class T2 > inline constexpr bool elements_are_additive_v = elements_are_additive< T1, T2 >::value;

// Test for subtractive elements
template < class T1, class T2, class = void > struct elements_are_subtractive : public ::std::false_type { };
template < class T1, class T2 > struct elements_are_subtractive< T1, T2, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< typename T1::value_type >() - ::std::declval< typename T2::value_type >() ), decltype( ::std::declval< typename T1::value_type >() - ::std::declval< typename T2::value_type >() ) > > > : public ::std::true_type { };
template < class T1, class T2 > inline constexpr bool elements_are_subtractive_v = elements_are_subtractive< T1, T2 >::value;

// Test for multiplicative elements
template < class T1, class T2, class = void > struct elements_are_multiplicative : public ::std::false_type { };
template < class T1, class T2 > struct elements_are_multiplicative< T1, T2, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< typename T1::value_type >() * ::std::declval< typename T2::value_type >() ), decltype( ::std::declval< typename T1::value_type >() * ::std::declval< typename T2::value_type >() ) > > > : public ::std::true_type { };
template < class T1, class T2 > inline constexpr bool elements_are_multiplicative_v = elements_are_multiplicative< T1, T2 >::value;

// Test for scalar pre-multiplicative
template < class S, class T, class = void > struct tensor_is_scalar_premultiplicative : public ::std::false_type { };
template < class S, class T > struct tensor_is_scalar_premultiplicative< S, T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< S >() * ::std::declval< typename T::value_type >() ), decltype( ::std::declval< S >() * ::std::declval< typename T::value_type >() ) > > > : public ::std::true_type { };
template < class S, class T > inline constexpr bool tensor_is_scalar_premultiplicative_v = tensor_is_scalar_premultiplicative< S, T >::value;

// Test for scalar post-multiplicative
template < class T, class S, class = void > struct tensor_is_scalar_postmultiplicative : public ::std::false_type { };
template < class T, class S > struct tensor_is_scalar_postmultiplicative< T, S, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< typename T::value_type >() * ::std::declval< S >() ), decltype( ::std::declval< typename T::value_type >() * ::std::declval< S >() ) > > > : public ::std::true_type { };
template < class T, class S > inline constexpr bool tensor_is_scalar_postmultiplicative_v = tensor_is_scalar_postmultiplicative< T, S >::value;

// Test for scalar divisible
template < class T, class S, class = void > struct tensor_is_scalar_divisible : public ::std::false_type { };
template < class T, class S > struct tensor_is_scalar_divisible< T, S, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< typename T::value_type >() / ::std::declval< S >() ), decltype( ::std::declval< typename T::value_type >() / ::std::declval< S >() ) > > > : public ::std::true_type { };
template < class T, class S > inline constexpr bool tensor_is_scalar_divisible_v = tensor_is_scalar_divisible< T, S >::value;

// Test for scalar modulo
template < class T, class S, class = void > struct tensor_is_scalar_modulo : public ::std::false_type { };
template < class T, class S > struct tensor_is_scalar_modulo< T, S, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval< typename T::value_type >() % ::std::declval< S >() ), decltype( ::std::declval< typename T::value_type >() % ::std::declval< S >() ) > > > : public ::std::true_type { };
template < class T, class S > inline constexpr bool tensor_is_scalar_modulo_v = tensor_is_scalar_modulo< T, S >::value;

// Test for matrices may be multiplicative
template < class M1, class M2, class = void > struct matrices_may_be_multiplicative : public ::std::false_type { };
template < class M1, class M2 > struct matrices_may_be_multiplicative< M1, M2, ::std::enable_if_t< ( M1::extents_type::static_extent(1) == M2::extents_type::static_extent(0) ) || ( M1::extents_type::static_extent(1) == ::std::dynamic_extent ) || ( M2::extents_type::static_extent(0) == ::std::dynamic_extent ) > > : public ::std::true_type { };
template < class M1, class M2 > inline constexpr bool matrices_may_be_multiplicative_v = matrices_may_be_multiplicative< M1, M2 >::value;

// Test for vector-matrix may be multiplicative
template < class V, class M, class = void > struct vector_matrix_may_be_multiplicative : public ::std::false_type { };
template < class V, class M > struct vector_matrix_may_be_multiplicative< V, M, ::std::enable_if_t< ( V::extents_type::static_extent(0) == M::extents_type::static_extent(0) ) || ( V::extents_type::static_extent(0) == ::std::dynamic_extent ) || ( M::extents_type::static_extent(0) == ::std::dynamic_extent ) > > : public ::std::true_type { };
template < class V, class M > inline constexpr bool vector_matrix_may_be_multiplicative_v = vector_matrix_may_be_multiplicative< V, M >::value;

// Test for matrix-vector may be multiplicative
template < class M, class V, class = void > struct matrix_vector_may_be_multiplicative : public ::std::false_type { };
template < class M, class V > struct matrix_vector_may_be_multiplicative< M, V, ::std::enable_if_t< ( V::extents_type::static_extent(0) == M::extents_type::static_extent(1) ) || ( V::extents_type::static_extent(0) == ::std::dynamic_extent ) || ( M::extents_type::static_extent(1) == ::std::dynamic_extent ) > > : public ::std::true_type { };
template < class M, class V > inline constexpr bool matrix_vector_may_be_multiplicative_v = matrix_vector_may_be_multiplicative< M, V >::value;

//- Test for tensors

// Tensor expression
template < class T > struct tensor_expression : public ::std::conditional_t< 
  has_value_type_v< T > &&
  has_size_type_v< T > &&
  has_extents_type_v< T > &&
  has_rank_type_v< T > &&
// #if LINALG_USE_BRACKET_OPERATOR
//   has_convertible_bracket_operator< T > &&
// #endif
// #if LINALG_USE_PAREN_OPERATOR
//   has_convertible_paren_operator< T > &&
// #endif
  has_rank_func_v< T > &&
  has_constexpr_rank_func_v< T > &&
  has_extents_func_v< T > &&
  has_extent_func_v< T >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool tensor_expression_v = tensor_expression< T >::value;

// Matrix expression
template < class T > struct matrix_expression : public ::std::conditional_t<
  tensor_expression_v< T > &&
  has_rank_v< T, 2 >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool matrix_expression_v = matrix_expression< T >::value;

// Vector expression
template < class T > struct vector_expression : public ::std::conditional_t<
  tensor_expression_v< T > &&
  has_rank_v< T, 1 >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool vector_expression_v = vector_expression< T >::value;

// Readable tensor
template < class T > struct readable_tensor : public ::std::conditional_t<
  tensor_expression_v< T > &&
  has_layout_type_v< T > &&
  has_mapping_type_v< T > &&
  has_accessor_type_v< T > &&
  has_reference_v< T > &&
  has_data_handle_type_v< T > &&
  has_rank_dynamic_func_v< T > &&
  has_static_extent_func_v< T > &&
  has_size_func_v< T > &&
  has_is_always_strided_func_v< T > &&
  has_is_always_exhaustive_func_v< T > &&
  has_is_always_unique_func_v< T > &&
  has_is_strided_func_v< T > &&
  has_is_exhaustive_func_v< T > &&
  has_is_unique_func_v< T > &&
  has_constexpr_rank_dynamic_func_v< T > &&
  has_constexpr_static_extent_func_v< T > &&
  has_constexpr_is_always_strided_func_v< T > &&
  has_constexpr_is_always_exhaustive_func_v< T > &&
  has_constexpr_is_always_unique_func_v< T > &&
  has_stride_func_v< T > &&
  has_accessor_func_v< T > &&
  has_data_handle_func_v< T > &&
  has_mapping_func_v< T >
  , ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool readable_tensor_v = readable_tensor< T >::value;

// Writable tensor
template < class T > struct writable_tensor : public ::std::conditional_t<
  readable_tensor_v< T >
// #if LINALG_USE_BRACKET_OPERATOR
//   && has_bracket_operator< T >
// #endif
// #if LINALG_USE_PAREN_OPERATOR
//   && has_paren_operator< T >
// #endif
  , ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool writable_tensor_v = writable_tensor< T >::value;

// Static tensor
template < class T > struct static_tensor : public ::std::conditional_t<
  writable_tensor_v< T > &&
  ( T::rank_dynamic() == 0 ) &&
  ::std::is_default_constructible_v< T >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool static_tensor_v = static_tensor< T >::value;

// Dynamic tensor
template < class T > struct dynamic_tensor : public ::std::conditional_t<
  writable_tensor_v< T > &&
  has_allocator_type_v< T > &&
  has_capacity_extents_type_v< T > &&
  has_max_size_func_v< T > &&
  has_capacity_func_v< T > &&
  has_resize_func_v< T > &&
  has_get_allocator_func_v< T > &&
  constructible_from_alloc_v< T > &&
  constructible_from_size_and_alloc_v< T >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool dynamic_tensor_v = dynamic_tensor< T >::value;

// Unevaluated tensor expression
template < class T > struct unevaluated_tensor_expression : public ::std::conditional_t<
  tensor_expression_v< T > &&
  has_evaluate_func_v< T >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool unevaluated_tensor_expression_v = unevaluated_tensor_expression< T >::value;

// Unary tensor expression
template < class T > struct unary_tensor_expression : public ::std::conditional_t<
  tensor_expression_v< T > &&
  has_underlying_func_v< T >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool unary_tensor_expression_v = unary_tensor_expression< T >::value;

// Binary tensor expression
template < class T > struct binary_tensor_expression : public ::std::conditional_t<
  unevaluated_tensor_expression_v< T > &&
  has_first_func_v< T > &&
  has_second_func_v< T >, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool binary_tensor_expression_v = binary_tensor_expression< T >::value;


#endif

LINALG_CONCEPTS_END // end concepts namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
