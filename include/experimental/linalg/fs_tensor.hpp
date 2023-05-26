//==================================================================================================
//  File:       fs_tensor.hpp
//
//  Summary:    This header defines a tensor - a multidimensional corrolary to std::array.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_FS_TENSOR_HPP
#define LINEAR_ALGEBRA_FS_TENSOR_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

/// @brief fs_tensor - a memory owning multidimensional container.
/// @tparam T type of element stored
/// @tparam Extents defines the multidimensional size
/// @tparam LayoutPolicy layout defines the ordering of elements in memory
/// @tparam AccessorPolicy accessor policy defines how elements are accessed
template < class T,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy >
class fs_tensor
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( Extents::rank_dynamic() == 0 )
#endif
{
  public:
    //- Types

    /// @brief Type returned by const index access must be convertible to this type
    using value_type               = T;
    /// @brief Type used to define memory layout
    using layout_type              = LayoutPolicy;
    /// @brief Type used to express size of tensor
    using extents_type             = Extents;
    /// @brief Type used to define access into memory
    using accessor_type            = AccessorPolicy;
    /// @brief Type used to define const access into memory
    using const_accessor_type      = detail::rebind_accessor_t< AccessorPolicy, const value_type >;
    /// @brief Type used for size along any dimension
    using size_type                = typename extents_type::size_type;
    // @brief Type used to express dimensions of the tensor
    using rank_type                = typename extents_type::rank_type;
    /// @brief Type returned by mutable index access
    using reference                = typename accessor_type::reference;
    /// @brief Type returned by const index access
    using const_reference          = ::std::add_const_t<typename accessor_type::reference>;
    /// @brief Type used to point to th beginning of the element buffer
    using data_handle_type         = typename accessor_type::data_handle_type;
    /// @brief Const type used to point to the beginning of the element buffer
    using const_data_handle_type   = const ::std::remove_pointer_t< typename accessor_type::data_handle_type > *;
    /// @brief Type used for indexing
    using index_type               = typename extents_type::index_type;

  private:

    //- Types

    /// @brief Type contained by the tensor
    using element_type             = typename accessor_type::element_type;
  
  public:

    //- Types

    /// @brief Type used to view the const memory within capacity
    using const_span_type          = ::std::experimental::mdspan< const element_type,
                                                                  extents_type,
                                                                  layout_type,
                                                                  LINALG_DETAIL::rebind_accessor_t< accessor_type, const element_type > >;
    /// @brief Type used to view the memory within capacity
    using span_type                = ::std::experimental::mdspan< element_type, extents_type, layout_type, accessor_type >;
    /// @brief Type use to map multidimensional indices into the buffer
    using mapping_type             = typename span_type::mapping_type;

    //- Destructor / Constructors / Assignments

    /// @brief Destructor
    LINALG_CONSTEXPR_DESTRUCTOR ~fs_tensor() noexcept( ::std::is_nothrow_destructible_v< element_type > );
    /// @brief Default constructor
    constexpr fs_tensor() noexcept( ::std::is_nothrow_default_constructible_v< element_type > ) = default;
    /// @brief Copy constructor
    /// @param rhs tensor to be copied
    constexpr fs_tensor( const fs_tensor& rhs ) noexcept( ::std::is_nothrow_copy_constructible_v< element_type > ) = default;
    /// @brief Construct from an initializer list
    /// @param il initializer list of elements to be copied
    explicit constexpr fs_tensor( const ::std::initializer_list<value_type>& il );
    /// @brief Constructs from an iterator pair
    /// @tparam InputIt Iterator Type
    /// @param first begin iterator
    /// @param last end iterator
    #ifdef LINALG_ENABLE_CONCEPTS
    template < ::std::input_iterator InputIt >
    constexpr fs_tensor( InputIt first, InputIt last );
    #endif
    #ifdef LINALG_RANGES_TO_CONTAINER
    /// @brief Construct from a range
    /// @tparam R range which satisfies input range concept
    /// @param tag range tag
    /// @param rg range
    #if LINALG_ENABLE_RANGES
    template < ::std::ranges::input_range R >
    #else
    template < class R >
    #endif
    constexpr fs_tensor( [[maybe_unused]] ::std::from_range_t, R&& rg );
    #endif
    /// @brief Construct by applying Tensor[indices...] to every element in the tensor
    /// @tparam Tensor tensor expression with an operator[]( indices ... ) defined
    /// @param t tensor expression to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < LINALG_CONCEPTS::tensor_expression Tensor >
    #else
    template < class Tensor,
               typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > &&
                                              ( Tensor::rank() == extents_type::rank() ) &&
                                              LINALG_DETAIL::extents_may_be_equal_v< extents_type,typename Tensor::extents_type > > >
    #endif
    explicit constexpr fs_tensor( Tensor&& t )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( Tensor::rank() == extents_type::rank() ) && LINALG_DETAIL::extents_may_be_equal_v< extents_type, typename Tensor::extents_type > )
    #endif
    ;
    /// @brief Copy assignment
    /// @param  fs_tensor to be copied
    /// @return self
    constexpr fs_tensor& operator = ( const fs_tensor& rhs );
    /// @brief Assign from an initializer list
    /// @param  il initializer list to be copied
    /// @return self
    constexpr fs_tensor& operator = ( const initializer_list<value_type>& il );
    /// @brief Assign from tensor expression
    /// @param tensor_expression to be copied
    /// @return self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < LINALG_CONCEPTS::tensor_expression Tensor >
    #else
    template < class Tensor,
               typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > &&
                                              ( Tensor::rank() == extents_type::rank() ) &&
                                              LINALG_DETAIL::extents_may_be_equal_v< extents_type,typename Tensor::extents_type > > >
    #endif
    constexpr fs_tensor& operator = ( Tensor&& rhs )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( Tensor::rank() == extents_type::rank() ) && LINALG_DETAIL::extents_may_be_equal_v< extents_type, typename Tensor::extents_type > )
    #endif
    ;

    //- Size / Capacity

    /// @brief Returns true if the tensor contains no elements
    /// @return bool
    [[nodiscard]] constexpr bool empty() const noexcept;
    /// @brief Returns the current number of (rows,columns,depth,etc.)
    /// @return number of (rows,columns,depth,etc.)
    [[nodiscard]] constexpr const extents_type& extents() const noexcept;
    /// @brief Returns the length of the tensor along the input dimension
    /// @return the length of the tensor along the input dimension
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept;
    /// @brief Returns the length of the tensor along the input dimension as known at compile time
    /// @return the length of the tensor along the input dimension as known at compile time
    [[nodiscard]] static constexpr size_type static_extent( rank_type n ) noexcept;
    /// @brief Returns the total number of elements contained
    /// @return the total number of elements contained
    [[nodiscard]] constexpr size_type size() const noexcept;
    /// @brief Returns the total number of elements the buffer may contain
    /// @return the total number of elements the buffer may contain
    [[nodiscard]] constexpr size_type max_size() const noexcept;

    //- Memory layout

    /// @brief true only if for every i and j where (i != j || ...) is true, m(i...) != m(j...) is true.
    /// @return bool
    [[nodiscard]] constexpr bool is_unique() const noexcept;
    /// @brief true only if for every i and j where (i != j || ...) is true, m(i...) != m(j...) is true.
    /// @return bool
    [[nodiscard]] constexpr bool is_exhaustive() const noexcept;
    /// @brief true only if for all k in the range [0, m.required_span_size() ) there exists an i such that m(i...) equals k. 
    /// @return bool
    [[nodiscard]] constexpr bool is_strided() const noexcept;
    /// @brief true if is_unique() returns true regardless of the dynamic state of the object.
    /// @return bool
    [[nodiscard]] static constexpr bool is_always_unique() noexcept;
    /// @brief true if is_exhaustive() returns true regardless of the dynamic state of the object.
    /// @return bool
    [[nodiscard]] static constexpr bool is_always_exhaustive() noexcept;
    /// @brief true if is_strided() returns true regardless of the dynamic state of the object.
    /// @return bool
    [[nodiscard]] static constexpr bool is_always_strided() noexcept;
    /// @brief The number of dimensions of the tensor
    /// @return rank
    [[nodiscard]] static constexpr rank_type rank() noexcept;
    /// @brief The number of dimensions of the tensor which are dynamic
    /// @return rank
    [[nodiscard]] static constexpr rank_type rank_dynamic() noexcept;
    /// @brief Returns the stride of the mapping along the input dimension
    /// @return the stride of the mapping along the input dimension
    [[nodiscard]] constexpr size_type stride( rank_type n ) const noexcept;
    /// @brief Returns the mapping object responsible for mapping indices into the memory buffer
    /// @return const reference to the mapping object
    [[nodiscard]] constexpr const mapping_type& mapping() const noexcept;

    //- Data access

    /// @brief Get a const pointer to the beginning of the element array
    /// @returns const data_handle_type
    [[nodiscard]] constexpr const_data_handle_type data_handle() const noexcept;
    /// @brief Get a pointer to the beginning of the element array
    /// @returns data_handle_type
    [[nodiscard]] constexpr data_handle_type data_handle() noexcept;
    /// @brief Returns the const accessor policy object
    /// @return the contained const accessor policy object
    [[nodiscard]] constexpr const_accessor_type accessor() const noexcept;
    /// @brief Returns the accessor policy object
    /// @return the contained accessor policy object
    [[nodiscard]] constexpr accessor_type accessor() noexcept;

    //- Const views

    /// @brief Returns the value at (indices...) without index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator[]( OtherIndexType ... indices ) const noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      ;
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( OtherIndexType ... indices ) const noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      ;
    #endif

    //- Mutable views

    /// @brief Returns a mutable value at (indices...) without index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns mutable value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr reference operator[]( OtherIndexType ... indices ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      ;
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr reference operator()( OtherIndexType ... indices ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      ;
    #endif

  private:
    //- Data
  
    /// @brief Accessor policy used
    [[no_unique_address]] accessor_type         accessor_;
    /// @brief Layout used for the current size
    [[no_unique_address]] mapping_type          size_map_;
    /// @brief Array of elements interpreted as a multidimensional array
    ::std::array< element_type, mapping_type().required_span_size() > elems_;

    //- Implementation details

    // Friend other tensor types
    template < class OtherT,
               class OtherExtents,
               class OtherLayoutPolicy,
               class OtherAccessorPolicy >
    friend class fs_tensor;
    
    // Attempts to copy view. If an exception is thrown, deallocates and rethrows
    template < class MDS >
    inline void copy_view_except( MDS&& span );
    // Calls destructor on all elements and deallocates the allocator
    // If an exception is thrown, the last exception to be thrown will be re-thrown.
    constexpr void destroy_all() noexcept( ::std::is_nothrow_destructible_v<element_type> );
    // Calls destructor on all elements and deallocates the allocator
    // If an exception is thrown, the last exception to be thrown will be re-thrown.
    inline void destroy_all_except();
    // Default constructs all elements
    // If an exception is to be thrown, will first destruct elements which have been constructed and deallocate
    constexpr void construct_all() noexcept( ::std::is_nothrow_constructible_v<element_type> );
    // Default constructs all elements
    // If an exception is to be thrown, will first destruct elements which have been constructed and deallocate
    inline void construct_all_except();
};

//-------------------------------------------------------
// Implementation of fs_tensor< T, Extents, LayoutPolicy, AccessorPolicy >
//-------------------------------------------------------

//- Destructor / Constructors / Assignments

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
LINALG_CONSTEXPR_DESTRUCTOR fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::~fs_tensor()
  noexcept( ::std::is_nothrow_destructible_v< typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::element_type > )
{
  this->destroy_all();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
constexpr fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::
fs_tensor( const ::std::initializer_list<value_type>& il ) :
  fs_tensor( il.begin(), il.end() )
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
template < ::std::input_iterator InputIt InputIt >
constexpr fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::
fs_tensor( InputIt first, InputIt last ) :
  acccessor_(),
  size_map_(),
  elems_( )
{
  if constexpr ( is_always_exhaustive() )
  {
    if constexpr ( ::std::contiguous_iterator<InputIt> )
    {
      if constexpr ( this->size_map_.required_span_size() == ( last - first ) )
      {
        LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                                 first,
                                 last,
                                 [&] ( const value_type& v ) constexpr noexcept
                                   { ::new( this->tm_.data() + sizeof(element_type) * ( ::std::addressof(v) - first ) / sizeof(value_type) ) element_type( v ); } );
      }
      else
      {
        // Size mismatch. Throw error.
        throw ::std::length_error("Initializer list size does not match required span size for the extents.");
      }
    }
    else
    {
      size_t count = 0;
      auto   iter  = first;
      while ( count < size_map_.required_span_size() )
      {
        ::new( this->tm_.data() + count ) element_type( *iter );
        ++iter;
        ++count;
      }
      if ( iter != last )
      {
        // Size mismatch. Throw error.
        throw ::std::length_error("Iterator pair size does not match required span size for the extents.");
      }
    }
  }
  else
  {
    static_assert( !is_always_exhaustive(), "Tensor does not support non-contiguous mapping types." );
  }
}
#endif

#ifdef LINALG_RANGES_TO_CONTAINER
template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
template < class R >
constexpr fs_tensor( [[maybe_unused]] ::std::from_range_t, R&& rg )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::ranges::input_range< R >
#endif
  :
  fs_tensor( rg.begin(), rg.end() )
{
}
#endif

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename >
#endif
constexpr fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::fs_tensor( Tensor&& t )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Tensor::rank() == tensor<T,Extents,LayoutPolicy,AccessorPolicy>::extents_type::rank() ) &&
             LINALG_DETAIL::extents_may_be_equal_v< typename tensor<T,Extents,LayoutPolicy,AccessorPolicy>::extents_type, typename Tensor::extents_type > )
#endif
  :
  accessor_(),
  size_map_( t.extents() ),
  elems_()
{
  // Construct all elements from tensor expression
  auto tensor_ctor = [this,&t]( auto ... indices ) constexpr noexcept( ::std::is_nothrow_copy_constructible_v<element_type> )
  {
    // TODO: This requires reference returned from mdspan to be the address of the element
    ::new ( ::std::addressof( LINALG_DETAIL::access( *this, indices ... ) ) ) element_type( LINALG_DETAIL::access( t, indices ... ) );
  };
  LINALG_DETAIL::apply_all( *this, tensor_ctor, LINALG_EXECUTION_UNSEQ );
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
constexpr fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>&
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::operator = ( const fs_tensor& rhs )
{
  if constexpr ( ! ::std::is_trivially_destructible_v< element_type > )
  {
    // Destroy all
    this->destroy_all();
  }
  if constexpr ( !LINALG_DETAIL::extents_are_equal_v< extents_type, typename fs_tensor::extents_type > )
  {
    if ( this->size_map_.extents() != rhs.extents() )
    {
      throw ::std::length_error("Extents must be equal");
    }
  }
  // Copy construct all elements
  LINALG_DETAIL::copy_view( this, rhs );
  
  return *this;
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
constexpr fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>&
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::operator = ( const initializer_list<value_type>& il )
{
  if constexpr ( is_always_exhaustive() )
  {
    if constexpr ( this->size_map_.required_span_size() == il.size() )
    {
      LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                               il.begin(),
                               il.end(),
                               [&] ( const value_type& v ) constexpr noexcept
                                 { *( this->elems_.data() + sizeof(element_type) * ( ::std::addressof(v) - il.begin() ) / sizeof(value_type) ) = v; } );
    }
    else
    {
      // Size mismatch. Throw error.
      throw ::std::length_error("Initializer list size does not match required span size for the extents.");
    }
  }
  else
  {
    static_assert( !is_always_exhaustive(), "Tensor does not support non-contiguous mapping types." );
  }
  return *this;
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename >
#endif
constexpr fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>&
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::operator = ( Tensor&& rhs )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Tensor::rank() == fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::extents_type::rank() ) &&
             LINALG_DETAIL::extents_may_be_equal_v< typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::extents_type, typename Tensor::extents_type > )
#endif
{
  if constexpr ( ! ::std::is_trivially_destructible_v< element_type > )
  {
    // Destroy all
    this->destroy_all();
  }
  if constexpr ( !LINALG_DETAIL::extents_are_equal_v< extents_type, typename Tensor::extents_type > )
  {
    if ( this->size_map_.extents() != rhs.extents() )
    {
      throw ::std::length_error("Extents must be equal");
    }
  }
  // Copy construct all elements
  LINALG_DETAIL::copy_view( this, rhs );
  
  return *this;
}

//- Size / Capacity

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr bool
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::empty() const noexcept
{
  return ( this->size() == 0 );
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr const typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::extents_type&
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::extents() const noexcept
{
  return this->size_map_.extents();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::size_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::extent( rank_type n ) const noexcept
{
  return this->size_map_.extents().extent( n );
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::size_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::static_extent( rank_type n ) noexcept
{
  return extents_type::static_extent( n );
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::size_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::size() const noexcept
{
  return this->size_map_.required_span_size();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::size_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::max_size() const noexcept
{
  return this->size();
}

//- Const views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::const_reference
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::operator[]( OtherIndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) &&
           ( ::std::is_convertible_v<OtherIndexType,typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::index_type> && ... )
#endif
{
  return this->accessor_.access( const_cast< data_handle_type >( this->elems_.data() ), this->size_map_( indices ... ) );
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::const_reference
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::operator()( OtherIndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::index_type> && ... )
#endif
{
  return this->accessor_.access( const_cast< data_handle_type >( this->elems_.data() ), this->size_map_( indices ... ) );
}
#endif

//- Mutable views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::reference
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::operator[]( OtherIndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::index_type> && ... )
#endif
{
  return this->accessor_.access( this->elems_.data(), this->size_map_( indices ... ) );
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::reference
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::operator()( OtherIndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::index_type> && ... )
#endif
{
  return this->accessor_.access( this->elems_.data(), this->size_map_( indices ... ) );
}
#endif

//- Memory layout

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr bool
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::is_unique() const noexcept
{
  return this->size_map_.is_unique();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr bool
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::is_exhaustive() const noexcept
{
  return this->size_map_.is_exhaustive();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr bool
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::is_strided() const noexcept
{
  return this->size_map_.is_strided();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr bool
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::is_always_unique() noexcept
{
  return mapping_type::is_always_unique();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr bool
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::is_always_exhaustive() noexcept
{
  return mapping_type::is_always_exhaustive();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr bool
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::is_always_strided() noexcept
{
  return mapping_type::is_always_strided();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::rank_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::rank() noexcept
{
  return extents_type::rank();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::rank_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::rank_dynamic() noexcept
{
  return extents_type::rank_dynamic();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::size_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::stride( rank_type n ) const noexcept
{
  return this->size_map_.stride( n );
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr const typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::mapping_type&
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::mapping() const noexcept
{
  return this->size_map_;
}

//- Data access

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::const_accessor_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::accessor() const noexcept
{
  return this->accessor_;
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::accessor_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::accessor() noexcept
{
  return this->accessor_;
}

//- Implementation detail

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
template < class MDS >
inline void fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::copy_view_except( MDS&& span )
{
  try
  {
    // Copy construct all elements
    LINALG_DETAIL::copy_view( *this, span );
  }
  catch ( ... )
  {
    // Rethrow
    rethrow_exception( current_exception() );
  }
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] inline constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::const_data_handle_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::data_handle() const noexcept
{
  return this->elems_.data();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
[[nodiscard]] inline constexpr typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::data_handle_type
fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::data_handle() noexcept
{
  return this->elems_.data();
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
constexpr void fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::destroy_all()
  noexcept( ::std::is_nothrow_destructible_v<typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::element_type> )
{
  // Is the destructor non-trivial?
  if constexpr ( ! ::std::is_trivially_destructible_v<element_type> )
  {
    if constexpr ( ::std::is_nothrow_destructible_v<element_type> )
    {
      if constexpr ( is_always_exhaustive() )
      {
        // If elements are contiguous, then just iterate over each in linear fashion.
        LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                                 this->data(),
                                 this->data() + this->size_map_.required_span_size(),
                                 []( const element_type& elem ) constexpr noexcept { elem.~element_type(); } );
      }
      else
      {
        // If elements are not-contiguous, then iterate using multidimensional indices
        LINALG_DETAIL::apply_all( this,
                                  [&]( auto ... indices ) constexpr noexcept
                                    { access( *this, indices ... ).~element_type(); },
                                  LINALG_EXECUTION_UNSEQ );
      }
    }
    else
    {
      this->destroy_all_except();
    }
  }
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
inline void fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::destroy_all_except()
{
  // Cache the last exception to be thrown
  ::std::exception_ptr eptr;
  if constexpr ( is_always_exhaustive() )
  {
    // Attempt to destruct in linear fashion
    LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                             this->data(),
                             this->data() + this->size_map_.required_span_size(),
                             [this,&eptr]( const element_type& elem ) { try { elem.~element_type(); } catch ( ... ) { eptr = ::std::current_exception(); } } );
  }
  else
  {
    // If elements are not-contiguous, then attempt to destroy using multidimensional indices
    LINALG_DETAIL::apply_all( this,
                              [&]( auto ... indices ) noexcept
                                { try { access( *this, indices ... ).~element_type(); } catch ( ... ) { eptr = ::std::current_exception(); } },
                              LINALG_EXECUTION_UNSEQ );
  }
  // If exceptions were thrown, rethrow the last
  if ( eptr ) LINALG_UNLIKELY
  {
    ::std::rethrow_exception( eptr );
  }
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
constexpr void fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::construct_all()
  noexcept( ::std::is_nothrow_constructible_v<typename fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::element_type> )
{
  if constexpr ( ::std::is_nothrow_constructible_v<element_type> )
  {
    // Construct without attempting to catch exception
    if constexpr ( is_always_exhaustive() )
    {
        // If elements are contiguous, then just iterate over each in linear fashion.
        LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                                 this->data(),
                                 this->data() + this->size_map_.required_span_size(),
                                 []( const element_type& elem ) constexpr noexcept { elem.element_type(); } );
    }
    else
    {
      // Iterate over the multi-index operator
      apply_all( *this,
                 [&]( auto ... indices ) constexpr noexcept
                   { ::new ( ::std::addressof( access( *this, indices ... ) ) ) element_type(); },
                 LINALG_EXECUTION_UNSEQ );
    }
  }
  else
  {
    this->construct_all_except();
  }
}

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
inline void fs_tensor<T,Extents,LayoutPolicy,AccessorPolicy>::construct_all_except()
{
  // Cache the last exception to be thrown
  ::std::exception_ptr eptr;
  // Attempt to construct
  if constexpr ( is_always_exhaustive() )
  {
    // Attempt to construct in linear fashion
    LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                             this->data(),
                             this->data() + this->size_map_.required_span_size(),
                             [this,&eptr]( const element_type& elem ) { try { elem.element_type(); } catch ( ... ) { eptr = ::std::current_exception(); } } );
  }
  else
  {
    // Attempt to construct via iteration over multi-index operator
    apply_all( *this,
                [&]( auto ... indices ) noexcept
                  { try { ::new ( ::std::addressof( access( *this, indices ... ) ) ) element_type(); } catch ( ... ) { eptr = ::std::current_exception(); } },
                LINALG_EXECUTION_UNSEQ );
  }
  // If exceptions were thrown, rethrow the last
  if ( eptr )
  {
    // Rethrow
    ::std::rethrow_exception( eptr );
  }
}

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_FS_TENSOR_HPP
