//==================================================================================================
//  File:       dr_tensor.hpp
//
//  Summary:    This header defines a dr_tensor - a memory owning dynamically sized
//              multidimensional container.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_DR_TENSOR_HPP
#define LINEAR_ALGEBRA_DR_TENSOR_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

/// @brief dr_tensor - a memory owning dynamically sized multidimensional container.
/// @tparam T type of element stored
/// @tparam Extents defines the multidimensional size
/// @tparam LayoutPolicy layout defines the ordering of elements in memory
/// @tparam CapExtents defines the capacity of the dr_tensor
/// @tparam Allocator allocator manages required memory
/// @tparam AccessorPolicy accessor policy defines how elements are accessed
template < class T,
           class Extents,
           class LayoutPolicy,
           class CapExtents,
           class Allocator,
           class AccessorPolicy >
class dr_tensor
{
  public:
    //- Types

    /// @brief Type returned by const index access must be convertible to this type
    using value_type               = T;
    /// @brief Type used to define memory layout
    using layout_type              = LayoutPolicy;
    /// @brief Type used to express size of dr_tensor
    using extents_type             = Extents;
    /// @brief Type used to express capacity of dr_tensor
    using capacity_extents_type    = CapExtents;
    /// @brief Type used to map multidimensional indices into the buffer
    using capacity_mapping_type    = typename layout_type::template mapping<capacity_extents_type>;
    /// @brief Type used to define access into memory
    using accessor_type            = AccessorPolicy;
    /// @brief Type used for size along any dimension
    using size_type                = typename extents_type::size_type;
    // @brief Type used to express dimensions of the dr_tensor
    using rank_type                = typename extents_type::rank_type;
    /// @brief Type returned by mutable index access
    using reference                = typename accessor_type::reference;
    /// @brief Type returned by const index access
    using const_reference          = ::std::add_const_t<typename accessor_type::reference>;
    /// @brief Type used to point to th beginning of the element buffer
    using data_handle_type         = typename accessor_type::data_handle_type;
    /// @brief Type used for indexing
    using index_type               = typename extents_type::index_type;
    /// @brief Type of allocator used to get memory
    using allocator_type           = Allocator;

  private:
    //- Types

    /// @brief Type contained by the dr_tensor
    using element_type             = typename accessor_type::element_type;
    /// @brief Type used to view the const memory within capacity
    using const_capacity_span_type = ::std::experimental::mdspan< const element_type,
                                                                  capacity_extents_type,
                                                                  layout_type,
                                                                  LINALG_DETAIL::rebind_accessor_t< accessor_type, const element_type > >;
    /// @brief Type used to view the memory within capacity
    using capacity_span_type       = ::std::experimental::mdspan< element_type, capacity_extents_type, layout_type, accessor_type >;

    //- Implementation details
    
    // Helper for defining the span type
    template < class MDS, class Seq >
    class span_impl;
    template < class MDS, auto ... Indices >
    class span_impl< MDS, index_sequence< Indices ... > >
    {
    private:
      template < class OtherIndex >
      [[nodiscard]] static inline constexpr auto full_ext( [[maybe_unused]] OtherIndex ) noexcept { return ::std::full_extent; }
    public:
      using type = decltype( ::std::experimental::submdspan( ::std::declval<capacity_span_type>(),
                                                             ::std::declval<decltype( full_ext( Indices ) )>() ... ) );
    };
    
  public:

    //- Types

    /// @brief Type used to view memory within size
    using span_type                = typename span_impl< capacity_span_type, ::std::make_index_sequence<extents_type::rank()> >::type;
    /// @brief Type used to const view memory within size
    using const_span_type          = typename span_impl< const_capacity_span_type, ::std::make_index_sequence<extents_type::rank()> >::type;
    /// @brief Type use to map multidimensional indices into the buffer
    using mapping_type             = typename span_type::mapping_type;

    //- Destructor / Constructors / Assignments

    /// @brief Destructor
    LINALG_CONSTEXPR_DESTRUCTOR ~dr_tensor() noexcept( ::std::is_nothrow_destructible_v<element_type> );

    /// @brief Default constructor
    constexpr dr_tensor() noexcept( ::std::is_nothrow_default_constructible_v<allocator_type> &&
                                    ( extents_type::rank_dynamic() != 0 ) );
    /// @brief Move constructor
    /// @param rhs dr_tensor to be moved
    constexpr dr_tensor( dr_tensor&& rhs ) noexcept;
    /// @brief Move constructor with allocator
    /// @param rhs dr_tensor to be moved
    /// @param alloc allocator to be used
    constexpr dr_tensor( dr_tensor&& rhs, const allocator_type& alloc );
    /// @brief Copy constructor
    /// @param rhs dr_tensor to be copied
    constexpr dr_tensor( const dr_tensor& rhs );
    /// @brief Copy constructor with allocator
    /// @param rhs dr_tensor to be copied
    /// @param alloc allocator to be used
    constexpr dr_tensor( const dr_tensor& rhs, const allocator_type& alloc );
    /// @brief Construct from an initializer list
    /// @param il initializer list of elements to be copied
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = ::std::enable_if_t< extents_type::rank_dynamic() == 0 > >
    #endif
    explicit constexpr dr_tensor( const ::std::initializer_list<value_type>& il, const allocator_type& alloc = allocator_type() );
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( extents_type::rank_dynamic() == 0 );
    #endif
    /// @brief Construct from an initializer list with a specified extents
    /// @param il initializer list of elements to be copied
    /// @param s  extents size
    constexpr dr_tensor( const ::std::initializer_list<value_type>& il, const extents_type& s, const allocator_type& alloc = allocator_type() );
    #ifdef LINALG_ENABLE_CONCEPTS
    /// @brief Constructs from an iterator pair
    /// @tparam InputIt Iterator Type
    /// @param first begin iterator
    /// @param last end iterator
    /// @param a allocator
    template < ::std::input_iterator InputIt >
    constexpr dr_tensor( InputIt first, InputIt last, const allocator_type& alloc = allocator_type() )
      requires ( ( extents_type::rank_dynamic() == 0 ) );
    /// @brief Constructs from an iterator pair
    /// @tparam InputIt Iterator Type
    /// @param first begin iterator
    /// @param last end iterator
    /// @param s extents size
    /// @param a allocator
    template < ::std::input_iterator InputIt >
    constexpr dr_tensor( InputIt first, InputIt last, const extents_type& s, const allocator_type& alloc = allocator_type() );
    #endif
    #ifdef LINALG_RANGES_TO_CONTAINER
    /// @brief Construct from a range
    /// @tparam R range which satisfies input range concept
    /// @param tag range tag
    /// @param rg range
    /// @param alloc allocator 
    template < class R >
    explicit constexpr dr_tensor( [[maybe_unused]] ::std::from_range_t, R&& rg, const allocator_type& alloc = allocator_type() )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::ranges::input_range< R >
    ;
    #endif
    /// @brief Construct from a range
    /// @tparam R range which satisfies input range concept
    /// @param tag range tag
    /// @param rg range
    /// @param s extents size
    /// @param alloc allocator 
    template < class R >
    constexpr dr_tensor( [[maybe_unused]] ::std::from_range_t, R&& rg, const extents_type& s, const allocator_type& alloc = allocator_type() )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::ranges::input_range< R >
    #endif
    ;
    #endif
    /// @brief Construct empty dimensionless dr_tensor with an allocator
    /// @param alloc allocator to construct with
    explicit constexpr dr_tensor( const allocator_type& alloc ) noexcept( extents_type::rank_dynamic() != 0 );
    /// @brief Attempt to allocate sufficient resources for a size dr_tensor and construct
    /// @param s defines the length of each dimension of the dr_tensor
    /// @param alloc allocator used to construct with
    explicit constexpr dr_tensor( extents_type s, const allocator_type& alloc = allocator_type() );
    /// @brief Construct by applying Tensor[indices...] to every element in the tensor
    /// @tparam Tensor tensor expression with an operator[]( indices ... ) defined
    /// @param tensor tensor expression to be performed on each element
    /// @param alloc allocator used to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < LINALG_CONCEPTS::tensor_expression Tensor >
    #else
    template < class Tensor,
               typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > &&
                                              ( Tensor::rank() == extents_type::rank() ) &&
                                              LINALG_DETAIL::extents_may_be_equal_v< extents_type,typename Tensor::extents_type > > >
    #endif
    explicit constexpr dr_tensor( Tensor&& t, const allocator_type& alloc = allocator_type() )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( Tensor::rank() == extents_type::rank() ) && LINALG_DETAIL::extents_may_be_equal_v< extents_type, typename Tensor::extents_type > )
    #endif
    ;
    /// @brief Move assignment
    /// @param dr_tensor to be moved
    /// @return self
    constexpr dr_tensor& operator = ( dr_tensor&& rhs )
      noexcept( typename ::std::allocator_traits<allocator_type>::propagate_on_container_move_assignment{} );
    /// @brief Copy assignment
    /// @param dr_tensor to be copied
    /// @return self
    constexpr dr_tensor& operator = ( const dr_tensor& rhs );
    /// @brief Assign from an initializer list
    /// @param  il initializer list to be copied
    /// @return self
    constexpr dr_tensor& operator = ( const initializer_list<value_type>& il );
    /// @brief Assign from dr_tensor expression
    /// @param dr_tensor_expression to be copied
    /// @return self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < LINALG_CONCEPTS::tensor_expression Tensor >
    #else
    template < class Tensor,
               typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > &&
                                              ( Tensor::rank() == extents_type::rank() ) &&
                                              LINALG_DETAIL::extents_may_be_equal_v< extents_type,typename Tensor::extents_type > > >
    #endif
    constexpr dr_tensor& operator = ( Tensor&& rhs )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( Tensor::rank() == extents_type::rank() ) && LINALG_DETAIL::extents_may_be_equal_v< extents_type, typename Tensor::extents_type > )
    #endif
    ;

    //- Size / Capacity

    /// @brief Returns true if the dr_tensor contains no elements
    /// @return bool
    [[nodiscard]] constexpr bool empty() const noexcept;
    /// @brief Returns the current number of (rows,columns,depth,etc.)
    /// @return number of (rows,columns,depth,etc.)
    [[nodiscard]] constexpr const extents_type& extents() const noexcept;
    /// @brief Returns the length of the dr_tensor along the input dimension
    /// @return the length of the dr_tensor along the input dimension
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept;
    /// @brief Returns the length of the dr_tensor along the input dimension as known at compile time
    /// @return the length of the dr_tensor along the input dimension as known at compile time
    [[nodiscard]] static constexpr size_type static_extent( rank_type n ) noexcept;
    /// @brief Returns the total number of elements contained
    /// @return the total number of elements contained
    [[nodiscard]] constexpr size_type size() const noexcept;
    /// @brief Returns the total number of elements the buffer may contain
    /// @return the total number of elements the buffer may contain
    [[nodiscard]] constexpr size_type max_size() const noexcept;
    /// @brief Returns the current capacity of (rows,columns,depth,etc.)
    /// @return capacity of (rows,columns,depth,etc.)
    [[nodiscard]] constexpr extents_type capacity() const noexcept;
    /// @brief Attempts to resize the dr_tensor to the input extents
    /// @param new_size extents type defining the new length of each dimension of the dr_tensor
    constexpr void resize( extents_type new_size );
    /// @brief Attempts to reserve the capacity of the dr_tensor to the input extents
    /// @param new_size extents type defining the new capacity along each dimension of the dr_tensor
    constexpr void reserve( extents_type new_cap );
    /// @brief Attempts to free up unused memory.
    constexpr void shrink_to_fit();

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
    /// @brief The number of dimensions of the dr_tensor
    /// @return rank
    [[nodiscard]] static constexpr rank_type rank() noexcept;
    /// @brief The number of dimensions of the dr_tensor which are dynamic
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
    [[nodiscard]] constexpr const data_handle_type data_handle() const noexcept;
    /// @brief Get a pointer to the beginning of the element array
    /// @returns data_handle_type
    [[nodiscard]] constexpr data_handle_type data_handle() noexcept;
    /// @brief returns the allocator being used
    /// @returns the allocator being used
    [[nodiscard]] constexpr allocator_type get_allocator() const noexcept;
    /// @brief Returns the accessor policy object
    /// @return the contained accessor policy object
    [[nodiscard]] constexpr const accessor_type& accessor() const noexcept;

    //- Const views

    /// @brief Returns the value at (indices...) without index bounds checking
    /// @param indices set indices representing a node in the dr_tensor
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
    /// @param indices set indices representing a node in the dr_tensor
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
    /// @brief Layout used for the current capacity
    [[no_unique_address]] capacity_mapping_type cap_map_;
    /// @brief Layout used for the current size
    [[no_unique_address]] mapping_type          size_map_;
    /// @brief Memory allocation and buffer pointer
    tensor_memory<element_type,allocator_type>  tm_;

    //- Implementation details

    // Friend other dr_tensor types
    template < class OtherT,
               class OtherExtents,
               class OtherLayoutPolicy,
               class OtherCapExtents,
               class OtherAllocator,
               class OtherAccessorPolicy >
    friend class dr_tensor;
    
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
    // Implementation of resize. (Needed a parameter pack of indices for implementation.)
    template < class SizeType, SizeType ... Indices >
    constexpr void resize_impl( extents_type new_size, [[maybe_unused]] ::std::integer_sequence<SizeType,Indices...> );
    // Returns an extents which is the maximum of the two inputs
    static constexpr extents_type max_extents( extents_type extents_a, extents_type extents_b ) noexcept;

};

//-------------------------------------------------------
// Implementation of dr_tensor< T, Extents, LayoutPolicy, CapExtents, Allocator, AccessorPolicy >
//-------------------------------------------------------

//- Destructor / Constructors / Assignments

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
LINALG_CONSTEXPR_DESTRUCTOR dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::~dr_tensor()
  noexcept( ::std::is_nothrow_destructible_v< typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::element_type > )
{
  // If the elements pointer has been set, then destroy and deallocate
  if ( this->tm_.data() ) LINALG_LIKELY
  {
    this->destroy_all();
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor()
  noexcept( ::std::is_nothrow_default_constructible_v< typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::allocator_type > &&
            ( extents_type::rank_dynamic() != 0 ) ) :
  dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>( allocator_type() )
{
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor( dr_tensor&& rhs ) noexcept :
  // Move accessor
  accessor_( ::std::move( rhs.accessor_ ) ),
  // Move capacity extents
  cap_map_( ::std::move( rhs.cap_map_ ) ),
  // Move mapping
  size_map_( ::std::move( rhs.size_map_ ) ),
  // Move memory
  tm_( ::std::move( rhs.tm_ ), this->cap_map_ )
{
  // Set the pointer in the moved dr_tensor to null so its destruction doesn't deallocate
  rhs.data_handle() = nullptr;
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor( dr_tensor&& rhs, const allocator_type& alloc ) :
  // Move accessor
  accessor_( ::std::move( rhs.accessor_ ) ),
  // Move capacity extents
  cap_map_( ::std::move( rhs.cap_map_ ) ),
  // Move mapping
  size_map_( ::std::move( rhs.size_map_ ) ),
  // Create memory
  tm_( alloc, this->cap_map_ )
{
  if constexpr ( ::std::is_nothrow_copy_constructible_v<element_type> )
  {
    // Copy construct all elements
    LINALG_DETAIL::copy_view( *this, rhs );
  }
  else
  {
    // Copy all elements - handling possible exceptions
    this->copy_view_except( rhs );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor( const dr_tensor& rhs ) :
  // Copy accessor
  accessor_( rhs.accessor_ ),
  // Copy capacity extents
  cap_map_( rhs.cap_map_ ),
  // Copy mapping
  size_map_( rhs.size_map_ ),
  // Copy memory
  tm_( rhs.tm_, this->cap_map_ )
{
  if constexpr ( ::std::is_nothrow_copy_constructible_v<element_type> )
  {
    // Copy construct all elements
    LINALG_DETAIL::copy_view( *this, rhs );
  }
  else
  {
    // Copy all elements - handling possible exceptions
    this->copy_view_except( rhs );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor( const dr_tensor& rhs, const allocator_type& alloc ) :
  // Copy accessor
  accessor_( rhs.accessor_ ),
  // Copy capacity extents
  cap_map_( rhs.cap_map_ ),
  // Copy mapping
  size_map_( rhs.size_map_ ),
  // Copy memory
  tm_( alloc, this->cap_map_ )
{
  if constexpr ( ::std::is_nothrow_copy_constructible_v<element_type> )
  {
    // Copy construct all elements
    LINALG_DETAIL::copy_view( *this, rhs );
  }
  else
  {
    // Copy all elements - handling possible exceptions
    this->copy_view_except( rhs );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
#ifndef LINALG_ENABLE_CONCEPTS
template < typename >
#endif
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::
dr_tensor( const ::std::initializer_list<value_type>& il, const allocator_type& alloc )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( extents_type::rank_dynamic() == 0 )
#endif
  :
  dr_tensor( il, extents_type(), alloc )
{
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::
dr_tensor( const ::std::initializer_list<value_type>& il, const extents_type& s, const allocator_type& alloc ) :
  accessor_(),
  cap_map_( s ),
  size_map_( s ),
  tm_( alloc, cap_map_ )
{
  if constexpr ( is_always_exhaustive() )
  {
    if constexpr ( size_map_.required_span_size() == il.size() )
    {
      LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                               il.begin(),
                               il.end(),
                               [&] ( const value_type& v ) constexpr noexcept
                                 { ::new( this->tm_.data() + sizeof(element_type) * ( ::std::addressof(v) - il.begin() ) / sizeof(value_type) ) element_type( v ); } );
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
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < ::std::input_iterator< InputIt > InputIt >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::
dr_tensor( InputIt first, InputIt last, const allocator_type& alloc  )
  requires ( ( extents_type::rank_dynamic() == 0 ) )
  :
  dr_tensor( first, last, extents_type(), alloc )
{
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < ::std::input_iterator< InputIt > InputIt >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::
dr_tensor( InputIt first, InputIt last, const extents_type& s, const allocator_type& alloc ) :
  accessor_(),
  cap_map_( s ),
  size_map_( s ),
  tm_( alloc, cap_map_ )
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
template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class R >
constexpr dr_tensor( [[maybe_unused]] ::std::from_range_t, R&& rg, const allocator_type& alloc )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::ranges::input_range< R >
#endif
  :
  dr_tensor( ::std::from_range_t(), rg, extents_type(), alloc )
{
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class R >
constexpr dr_tensor( [[maybe_unused]] ::std::from_range_t, R&& rg, const extents_type& s, const allocator_type& alloc )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::ranges::input_range< R >
#endif
  :
  accessor_(),
  cap_map_( s ),
  size_map_( s ),
  tm_( alloc, cap_map_ )
{
  if constexpr ( is_always_exhaustive() )
  {
    size_t count = 0;
    auto iter = ::std::ranges::begin(rg);
    while ( count < size_map_.required_span_size() )
    {
      ::new( this->tm_.data() + count ) element_type( *iter );
      ++iter;
      ++count;
    }
    if ( iter != ::std::ranges::end(rg) )
    {
      // Size mismatch. Throw error.
      throw ::std::length_error("Iterator pair size does not match required span size for the extents.");
    }
  }
  else
  {
    static_assert( !is_always_exhaustive(), "Tensor does not support non-contiguous mapping types." );
  }
}
#endif

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor( const allocator_type& alloc ) noexcept( extents_type::rank_dynamic() != 0 ) :
  accessor_(),
  cap_map_(),
  size_map_( this->cap_map_ ),
  tm_( alloc, this->cap_map_ )
{
  if constexpr ( LINALG_DETAIL::extents_is_static_v<extents_type> &&
                 !( ::std::is_trivially_default_constructible_v<element_type> &&
                    ::std::is_trivially_copy_assignable_v<element_type> &&
                    ::std::is_trivially_destructible_v<element_type> ) )
  {
    this->construct_all();
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor( extents_type s, const allocator_type& alloc ) :
  accessor_(),
  cap_map_( s ),
  size_map_( this->cap_map_ ),
  tm_( alloc, this->cap_map_ )
{
  // If construct, assign, and destruct are not trivial, then initialize data
  if constexpr ( !( ::std::is_trivially_default_constructible_v<element_type> &&
                    ::std::is_trivially_copy_assignable_v<element_type> &&
                    ::std::is_trivially_destructible_v<element_type> ) )
  {
    this->construct_all();
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename >
#endif
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::dr_tensor( Tensor&& t, const allocator_type& alloc )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Tensor::rank() == dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents_type::rank() ) &&
             LINALG_DETAIL::extents_may_be_equal_v< typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents_type, typename Tensor::extents_type > )
#endif
  :
  accessor_(),
  cap_map_( t.extents() ),
  size_map_( this->cap_map_ ),
  tm_( alloc, this->cap_map_ )
{
  // Construct all elements from tensor expression
  auto tensor_ctor = [this,&t]( auto ... indices ) constexpr noexcept( ::std::is_nothrow_copy_constructible_v<element_type> )
  {
    // TODO: This requires reference returned from mdspan to be the address of the element
    ::new ( ::std::addressof( LINALG_DETAIL::access( *this, indices ... ) ) ) element_type( LINALG_DETAIL::access( t, indices ... ) );
  };
  LINALG_DETAIL::apply_all( *this, tensor_ctor, LINALG_EXECUTION_UNSEQ );
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>&
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator = ( dr_tensor&& rhs )
  noexcept( typename ::std::allocator_traits< typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::allocator_type >::propagate_on_container_move_assignment{} )
{
  // If the allocator is moved, then move everything
  if constexpr ( typename ::std::allocator_traits<allocator_type>::propagate_on_container_move_assignment {} )
  {
    // Destroy everything currently allocated
    if ( this->tm_.data() ) LINALG_LIKELY
    {
      this->destroy_all();
    }
    // Move
    this->cap_map_  = ::std::move( rhs.cap_map_ );
    this->size_map_ = ::std::move( rhs.size_map_ );
    this->tm_       = ::std::move( rhs.tm_ );
    // Set moved tensor element pointer to null so its destruction doesn't deallocate
    this->data() = nullptr;
  }
  else
  {
    if constexpr ( ::std::is_trivially_destructible_v<element_type> )
    {
      if ( this->capacity() != rhs.capacity() )
      {
        // Deallocate
        this->tm_.deallocate( this->cap_map_ );
        // Set new capacity
        this->cap_map_  = rhs.cap_map_;
        // Set new size
        this->size_map_ = rhs.size_map_;
        // Set memory
        this->tm_.alloocate( this->cap_ );
        // Copy construct all elements
        LINALG_DETAIL::copy_view( *this, rhs );
      }
      else
      {
        // Set new size
        this->size_map_ = rhs.size_map_;
        // Copy construct all elements
        LINALG_DETAIL::copy_view( *this, rhs );
      }
    }
    else
    {
      // Destroy all elements
      this->destroy_all();
      // Set new capacity
      this->cap_map_  = rhs.cap_map_;
      // Set new size
      this->size_map_ = rhs.size_map_;
      // Set memory
      this->tm_.alloocate( this->cap_map_ );
      // Copy construct all elements
      LINALG_DETAIL::copy_view( *this, rhs );
    }
  }
  return *this;
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>&
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator = ( const dr_tensor& rhs )
{
  if constexpr ( ::std::is_trivially_destructible_v<element_type> )
  {
    if ( this->cap_map_ != rhs.cap_map_ )
    {
      // Deallocate
      this->tm_.deallocate( this->cap_map_ );
      // Propogate allocator
      tm_.assign_allocator( rhs.tm_ );
      // Set new capacity
      this->cap_map_ = rhs.cap_map_;
      // Copy construct all elements
      LINALG_DETAIL::copy_view( this, rhs );
    }
    else
    {
      if constexpr ( typename ::std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment{} )
      {
        // Deallocate
        this->tm_.deallocate( this->cap_map_ );
        // Propogate allocator
        tm_.assign_allocator( rhs.tm_ );
      }
      // Set new size
      this->size_map_ = rhs.size_map_;
      // Copy construct all elements
      LINALG_DETAIL::copy_view( this, rhs );
    }
  }
  else
  {
    // Destroy all
    this->destroy_all();
    // Propogate allocator
    tm_.assign_allocator( rhs.tm_ );
    // Set new capacity
    this->cap_map_  = rhs.cap_map_;
    // Set new size
    this->size_map_ = rhs.size_map_;
    // Copy construct all elements
    LINALG_DETAIL::copy_view( this, rhs );
  }
  return *this;
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>&
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator = ( const initializer_list<value_type>& il )
{
  if constexpr ( is_always_exhaustive() )
  {
    if constexpr ( this->size_map_.required_span_size() == il.size() )
    {
      LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                              il.begin(),
                              il.end(),
                              [&] ( const value_type& v ) constexpr noexcept
                                { *( this->tm_.data() + sizeof(element_type) * ( ::std::addressof(v) - il.begin() ) / sizeof(value_type) ) = v; } );
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

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename >
#endif
constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>&
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator = ( Tensor&& rhs )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Tensor::rank() == dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents_type::rank() ) && LINALG_DETAIL::extents_may_be_equal_v< typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents_type, typename Tensor::extents_type > )
#endif
{
  if constexpr ( ::std::is_trivially_destructible_v<element_type> )
  {
    if ( this->cap_map_.extents() != rhs.extents() )
    {
      // Deallocate
      this->tm_.deallocate( this->cap_map_ );
      // Set new capacity
      this->cap_map_ = capacity_mapping_type( rhs.extents() );
      // Allocate
      this->tm_.allocate( this->cap_map_ );
      // Copy construct all elements
      LINALG_DETAIL::copy_view( this, rhs );
    }
    else
    {
      // Set new size
      this->size_map_ = mapping_type( rhs.extents() );
      // Copy construct all elements
      LINALG_DETAIL::copy_view( this, rhs );
    }
  }
  else
  {
    // Destroy all
    this->destroy_all();
    // Set new capacity
    this->cap_map_ = capacity_mapping_type( rhs.extents() );
    // Allocate
    this->tm_.allocate( this->cap_map_ );
    // Set new size
    this->size_map_ = mapping_type( rhs.extents() );
    // Copy construct all elements
    LINALG_DETAIL::copy_view( this, rhs );
  }
  return *this;
}

//- Size / Capacity

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr bool
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::empty() const noexcept
{
  return ( this->size() == 0 );
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr const typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents_type&
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents() const noexcept
{
  return this->size_map_.extents();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::size_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extent( rank_type n ) const noexcept
{
  return this->size_map_.extents().extent( n );
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::size_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::static_extent( rank_type n ) noexcept
{
  return extents_type::static_extent( n );
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::size_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::size() const noexcept
{
  return this->size_map_.required_span_size();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::size_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::max_size() const noexcept
{
  return ::std::allocator_traits<allocator_type>::max_size( this->alloc_ );
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::capacity() const noexcept
{
  return this->cap_;
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::resize( extents_type new_size )
{
  // Check if the memory layout must change
  if ( LINALG_DETAIL::sufficient_extents( this->cap_, new_size ) )
  {
    this->resize_impl( new_size, ::std::make_integer_sequence<index_type,extents_type::rank()>() );
  }
  else
  {
    // Copy current state
    dr_tensor clone = ::std::move( *this );
    // Set to new size
    *this = dr_tensor( new_size, max_extents( new_size, this->capacity() ), this->get_allocator() );
    // Copy view
    LINALG_DETAIL::assign_view( *this, clone );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::reserve( extents_type new_cap )
{
  // Only expand if capacity is not currently sufficient
  if ( !LINALG_DETAIL::sufficient_extents( this->cap_, new_cap ) )
  {
    // Copy current state
    dr_tensor clone = ::std::move( *this );
    // Set to new size
    *this = dr_tensor( this->size(), max_extents( new_cap, this->capacity() ), this->get_allocator() );
    // Copy view
    LINALG_DETAIL::assign_view( *this, clone );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::shrink_to_fit()
{
  // Copy current state
  dr_tensor clone = ::std::move( *this );
  // Set to new dr_tensor with size and capacity matching
  *this = dr_tensor( this->size(), this->get_allocator() );
  // Copy view
  LINALG_DETAIL::assign_view( *this, clone );
}

//- Const views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::const_reference
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator[]( OtherIndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) &&
           ( ::std::is_convertible_v< OtherIndexType,typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::index_type > && ... )
#endif
{
  return this->accessor_( this->tm_.data(), this->size_map_( indices ... ) );
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::const_reference
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator()( OtherIndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::index_type> && ... )
#endif
{
  return this->accessor_( this->tm_.data(), this->size_map_( indices ... ) );
}
#endif

//- Mutable views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::reference
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator[]( OtherIndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v< OtherIndexType,typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::index_type > && ... )
#endif
{
  return this->accessor_( this->tm_.data(), this->size_map_( indices ... ) );
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class ... OtherIndexType >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::reference
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::operator()( OtherIndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::index_type> && ... )
#endif
{
  return this->accessor_( this->tm_.data(), this->size_map_( indices ... ) );
}
#endif

//- Memory layout

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr bool
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::is_unique() const noexcept
{
  return this->size_map_.is_unique();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr bool
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::is_exhaustive() const noexcept
{
  return this->size_map_.is_exhaustive();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr bool
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::is_strided() const noexcept
{
  return this->size_map_.is_strided();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr bool
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::is_always_unique() noexcept
{
  return mapping_type::is_always_unique();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr bool
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::is_always_exhaustive() noexcept
{
  return mapping_type::is_always_exhaustive();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr bool
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::is_always_strided() noexcept
{
  return mapping_type::is_always_strided();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::rank_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::rank() noexcept
{
  return extents_type::rank();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::rank_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::rank_dynamic() noexcept
{
  return extents_type::rank_dynamic();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::size_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::stride( rank_type n ) const noexcept
{
  return this->size_map_.stride( n );
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr const typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::mapping_type&
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::mapping() const noexcept
{
  return this->size_map_;
}

//- Data access

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::allocator_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::get_allocator() const noexcept
{
  return this->tm_.get_allocator();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] constexpr const typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::accessor_type&
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::accessor() const noexcept
{
  return this->accessor_;
}

//- Implementation detail


template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class MDS >
inline void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::copy_view_except( MDS&& span )
{
  try
  {
    // Copy construct all elements
    LINALG_DETAIL::copy_view( *this, span );
  }
  catch ( ... )
  {
    // Deallocate
    this->tm_.deallocate( this->cap_map_ );
    // Rethrow
    rethrow_exception( current_exception() );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] inline constexpr const typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::data_handle_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::data_handle() const noexcept
{
  return this->tm_.data();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
[[nodiscard]] inline constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::data_handle_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::data_handle() noexcept
{
  return this->tm_.data();
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::destroy_all()
  noexcept( ::std::is_nothrow_destructible_v<typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::element_type> )
{
  // Is the destructor non-trivial?
  if constexpr ( ::std::is_trivially_destructible_v<element_type> )
  {
    // Deallocate
    this->tm_.deallocate( this->cap_map_ );
  }
  else
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
      // Deallocate
      this->tm_.deallocate( this->cap_map_ );
    }
    else
    {
      this->destroy_all_except();
    }
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
inline void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::destroy_all_except()
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
  // Deallocate
  this->tm_.deallocate( this->cap_map_ );
  // If exceptions were thrown, rethrow the last
  if ( eptr ) LINALG_UNLIKELY
  {
    ::std::rethrow_exception( eptr );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::construct_all()
  noexcept( ::std::is_nothrow_constructible_v<typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::element_type> )
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

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
inline void dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::construct_all_except()
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
    // Deallocate
    this->tm_.deallocate( this->cap_map_ );
    // Rethrow
    ::std::rethrow_exception( eptr );
  }
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
template < class SizeType, SizeType ... Indices >
constexpr void
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::
resize_impl( extents_type     new_size,
             [[maybe_unused]] ::std::integer_sequence<SizeType,Indices...> )
{
  // If not trivially destructible, then elements descoped from resize must be deleted
  // and elements added to scope must be default constructed
  if constexpr ( !::std::is_trivially_destructible_v<element_type> )
  {
    // Create subview of elements to be destroyed
    auto destroy_extent = [this,new_size]( SizeType index ) constexpr noexcept
    {
      return this->size().extent(index) > new_size(index) ?
               tuple( new_size.extent(index), this->size().extent(index) ) :
               tuple( this->size().extent(index), this->size().extent(index) );
    };
    auto destroy_subview = ::std::experimental::submdspan( span_type( this->data_handle(), this->mapping(), this->accessor() ),
                                                           destroy_extent(Indices) ...  );
    // Define destructor lambda
    auto destructor = [this]( auto ... indices ) constexpr noexcept( ::std::is_nothrow_destructible_v<element_type> )
      { LINALG_DETAIL::access( *this, indices ... ).~element_type(); };
    // Destroy
    LINALG_DETAIL::apply_all( destroy_subview, destructor, LINALG_EXECUTION_UNSEQ );
    // Create subview of elements to be constructed
    auto construct_extent = [this,new_size]( SizeType index ) constexpr noexcept
    {
      return this->size().extent(index) < new_size(index) ?
               tuple( this->size().extent(index), new_size.extent(index) ) :
               tuple( this->size().extent(index), this->size().extent(index) );
    };
    auto construct_subview = ::std::experimental::submdspan( span_type( this->data_handle(), this->mapping(), this->accessor() ),
                                                             construct_extent(Indices) ...  );
    // Define constructor lambda
    auto constructor = [this]( auto ... indices ) constexpr noexcept( ::std::is_nothrow_default_constructible_v<element_type> )
      { ::new ( ::std::addressof( LINALG_DETAIL::access( *this, indices ... ) ) ) element_type(); };
    // Construct
    LINALG_DETAIL::apply_all( construct_subview, constructor, LINALG_EXECUTION_UNSEQ );
  }
  // Create a new mapping
  this->size_map_ = mapping_type( new_size );
}

template < class T, class Extents, class LayoutPolicy, class CapExtents, class Allocator, class AccessorPolicy >
constexpr typename dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::extents_type
dr_tensor<T,Extents,LayoutPolicy,CapExtents,Allocator,AccessorPolicy>::
max_extents( extents_type extents_a, extents_type extents_b ) noexcept
{
  // Construct array to contain max
  ::std::array<index_type,extents_type::rank()> max_extents;
  // Iterate over each dimension and set max
  LINALG_DETAIL::for_each( LINALG_EXECUTION_UNSEQ,
                           detail::faux_index_iterator<index_type>( 0 ),
                           detail::faux_index_iterator<index_type>( extents_type::rank() ),
                           [&max_extents,&extents_a,&extents_b] ( index_type index ) constexpr noexcept
                           {
                             max_extents[index] = ( extents_a.extent(index) > extents_b.extent(index) ) ? extents_a.extent(index) : extents_b.extent(index);
                           } );
  // Return max
  return extents_type( max_extents );
}

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_DR_TENSOR_HPP
