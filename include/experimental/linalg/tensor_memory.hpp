//==================================================================================================
//  File:       tensor_memory.hpp
//
//  Summary:    This header defines a tensor memory which wraps buffer maintenance for a tensor
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_MEMORY_HPP
#define LINEAR_ALGEBRA_TENSOR_MEMORY_HPP

#include <experimental/linear_algebra.hpp>

// forward declaration
namespace util
{
template < class T, ::std::size_t N >
class fixed_size_allocator;
}

namespace std
{
namespace experimental
{

template < class T,
           class Allocator >
class tensor_memory
{
  private:
    // Helper class for defining allocator behavior
    template < class U, class propogate_on_copy_true >
    struct Alloc_copy_helper { [[nodiscard]] static inline constexpr auto propogate( const U& u ) noexcept { return u.get_allocator(); } };
    template < class U >
    struct Alloc_copy_helper< U, false_type > { [[nodiscard]] static inline constexpr auto propogate( [[maybe_unused]] const U& u ) noexcept { return ::std::move( allocator_type() ); } };
    // Helper class for defining allocator behavior
    template < class U, class propogate_on_move_true >
    struct Alloc_move_helper { [[nodiscard]] static inline constexpr auto propogate( U&& u ) noexcept { return ::std::move( u.get_allocator() ); } };
    template < class U >
    struct Alloc_move_helper< U, false_type > { [[nodiscard]] static inline constexpr auto propogate( [[maybe_unused]] U&& u ) noexcept { return ::std::move( allocator_type() ); } };
  public:
    //- Types

    using element_type           = T;
    using allocator_type         = Allocator;
    using pointer                = typename ::std::allocator_traits<allocator_type>::pointer;
    using const_pointer          = typename ::std::allocator_traits<allocator_type>::const_pointer;

    //- Destructor / Constructors / Assignments

    // Destructor
    constexpr ~tensor_memory() noexcept = default;
    // Default constructor
    constexpr tensor_memory() noexcept requires ( ::std::is_default_constructible_v<allocator_type> ) = default;
    // Template copy construction
    template < class U, class Alloc, class MappingType >
    constexpr tensor_memory( const tensor_memory<U,Alloc>& tm, const MappingType& mapping );
    // Template move construction
    template < class MappingType >
    constexpr tensor_memory( tensor_memory&& tm, const MappingType& mapping ) noexcept;
    // Construct from allocator (no allocation)
    template < class Alloc >
    constexpr tensor_memory( const Alloc& alloc ) noexcept;
    // Construct from allocator. Allocate to support size of mapping.
    template < class Alloc, class MappingType >
    constexpr tensor_memory( const Alloc& alloc, const MappingType& mapping );
    // Move assignment
    constexpr tensor_memory& operator = ( tensor_memory&& tm ) noexcept;
    // Assign allocator
    template < class U, class Alloc >
    constexpr void assign_allocator( const tensor_memory<U,Alloc>& tm ) noexcept;

    //- Data access

    // Const data pointer
    [[nodiscard]] constexpr const_pointer data() const noexcept;
    // Data pointer
    [[nodiscard]] constexpr pointer data() noexcept;
    // Copy of allocator
    [[nodiscard]] constexpr allocator_type get_allocator() const noexcept;

    //- Memory functions

    // Allocate
    template < class MappingType >
    constexpr void allocate( const MappingType& mapping );
    // Deallocate
    template < class MappingType >
    constexpr void deallocate( const MappingType& mapping );
  private:

    //- Implementation detail

    // Alias for allocator type actually used
    using rebound_allocator_type = typename ::std::allocator_traits<allocator_type>::template rebind_alloc<element_type>;

    // Friend access
    template < class U, class Alloc >
    friend class tensor_memory;

    //- Data

    // Contained allocator used to manage memory
    [[no_unique_address]] rebound_allocator_type alloc_;
    // Pointer to beginning of memory buffer
    pointer p_;
};

//- Destructor / Constructors / Assignments

template < class T, class Allocator >
template < class U, class Alloc, class MappingType >
constexpr tensor_memory<T,Allocator>::
tensor_memory( const tensor_memory<U,Alloc>& tm, const MappingType& mapping ) :
  // Copy or default construct allocator
  alloc_( ::std::allocator_traits<typename tensor_memory<U,Alloc>::rebound_allocator_type>::select_on_container_copy_construction( tm.alloc_ ) ),
  // Allocate elements
  p_( ::std::allocator_traits<typename tensor_memory<T,Allocator>::rebound_allocator_type>::allocate( this->alloc_, mapping.required_span_size() ) )
{
}

template < class T, class Allocator >
template < class MappingType >
constexpr tensor_memory<T,Allocator>::
tensor_memory( tensor_memory&& tm, const MappingType& mapping ) noexcept :
  // Move or construct allocator
  alloc_( ::std::move( tm.alloc_ ) ),
  // Move pointer
  p_( ::std::move( tm.p_ ) )
{
  tm.p_ = nullptr;
}

template < class T, class Allocator >
template < class Alloc >
constexpr tensor_memory<T,Allocator>::tensor_memory( const Alloc& alloc ) noexcept :
  // Copy allocator
  alloc_( alloc ),
  // Set pointer to null
  p_( nullptr )
{
}

template < class T, class Allocator >
template < class Alloc, class MappingType >
constexpr tensor_memory<T,Allocator>::tensor_memory( const Alloc& alloc, const MappingType& mapping ) :
  // Copy allocator
  alloc_( alloc ),
  // Allocate buffer
  p_( ::std::allocator_traits<rebound_allocator_type>::allocate( this->alloc_, mapping.required_span_size() ) )
{
}

template < class T, class Allocator >
constexpr tensor_memory<T,Allocator>& tensor_memory<T,Allocator>::operator = ( tensor_memory&& tm ) noexcept
{
  this->alloc_ = ::std::move( tm.alloc_ );
  this->p_     = ::std::move( tm.p_ );
  tm.p         = nullptr;
}

template < class T, class Allocator >
template < class U, class Alloc >
constexpr void tensor_memory<T,Allocator>::assign_allocator( const tensor_memory<U,Alloc>& tm ) noexcept
{
  if constexpr ( typename ::std::allocator_traits<rebound_allocator_type>::propagate_on_container_copy_assignment{} )
  {
    this->alloc_ = tm.alloc_;
  }
}

//- Data access

template < class T, class Allocator >
[[nodiscard]] constexpr typename tensor_memory<T,Allocator>::const_pointer
tensor_memory<T,Allocator>::data() const noexcept
{
  return this->p_;
}

template < class T, class Allocator >
[[nodiscard]] constexpr typename tensor_memory<T,Allocator>::pointer
tensor_memory<T,Allocator>::data() noexcept
{
  return this->p_;
}

// Copy of allocator
template < class T, class Allocator >
[[nodiscard]] constexpr typename tensor_memory<T,Allocator>::allocator_type
tensor_memory<T,Allocator>::get_allocator() const noexcept
{
  return this->alloc_;
}

//- Memory functions

template < class T, class Allocator >
template < class MappingType >
constexpr void
tensor_memory<T,Allocator>::allocate( const MappingType& mapping )
{
  this->p_ = ::std::allocator_traits<rebound_allocator_type>::allocate( this->alloc_, mapping.required_span_size() );
}

template < class T, class Allocator >
template < class MappingType >
constexpr void
tensor_memory<T,Allocator>::deallocate( const MappingType& mapping )
{
  ::std::allocator_traits<rebound_allocator_type>::deallocate( this->alloc_, this->p_, mapping.required_span_size() );
}



// Specialization for self contained memory
template < class         T,
           ::std::size_t N >
class tensor_memory< T, util::fixed_size_allocator<T,N> >
{
  public:
    //- Types

    using element_type   = T;
    using allocator_type = util::fixed_size_allocator<T,N>;
    using pointer        = typename ::std::allocator_traits<allocator_type>::pointer;
    using const_pointer  = typename ::std::allocator_traits<allocator_type>::const_pointer;

    //- Destructor / Constructors / Assignments

    // Destructor
    constexpr ~tensor_memory() noexcept = default;
    // Default constructor
    constexpr tensor_memory() noexcept = default;
    // Template copy construction
    template < class U, class Alloc, class MappingType >
    constexpr tensor_memory( [[maybe_unused]] const tensor_memory<U,Alloc>& tm, [[maybe_unused]] const MappingType& mapping ) noexcept;
    // Template move construction
    template < class MappingType >
    constexpr tensor_memory( [[maybe_unused]] tensor_memory&& tm, [[maybe_unused]] const MappingType& mapping ) noexcept;
    // Construct from allocator (no allocation)
    template < class Alloc >
    constexpr tensor_memory( [[maybe_unused]] const Alloc& alloc ) noexcept;
    // Construct from allocator. Allocate to support size of mapping.
    template < class Alloc, class MappingType >
    constexpr tensor_memory( [[maybe_unused]] const Alloc& alloc, [[maybe_unused]] const MappingType& mapping ) noexcept;
    // Move assignment
    constexpr tensor_memory& operator = ( [[maybe_unused]] tensor_memory&& tm ) noexcept;
    // Assign allocator
    template < class U, class Alloc >
    constexpr void assign_allocator( [[maybe_unused]] const tensor_memory<U,Alloc>& tm ) noexcept;

    //- Data access

    // Const data pointer
    [[nodiscard]] constexpr const_pointer data() const noexcept;
    // Data pointer
    [[nodiscard]] constexpr pointer data() noexcept;
    // Get copy of allocator
    [[nodiscard]] constexpr allocator_type get_allocator() const noexcept;

    //- Memory functions

    // Allocate
    template < class MappingType >
    constexpr void allocate( [[maybe_unused]] const MappingType& mapping );
    // Deallocate
    template < class MappingType >
    constexpr void deallocate( [[maybe_unused]] const MappingType& mapping );

  private:

    //- Data

    // Contained allocator used to manage memory
    allocator_type alloc_;
};

//- Destructor / Constructors / Assignments

template < class T, ::std::size_t N >
template < class U, class Alloc, class MappingType >
constexpr tensor_memory< T, util::fixed_size_allocator<T,N> >::
tensor_memory( [[maybe_unused]] const tensor_memory<U,Alloc>& tm, [[maybe_unused]] const MappingType& mapping ) noexcept :
  alloc_()
{
}

template < class T, ::std::size_t N >
template < class MappingType >
constexpr tensor_memory< T, util::fixed_size_allocator<T,N> >::
tensor_memory( [[maybe_unused]] tensor_memory&& tm, [[maybe_unused]] const MappingType& mapping ) noexcept :
  alloc_()
{
}

template < class T, ::std::size_t N >
template < class Alloc >
constexpr tensor_memory< T, util::fixed_size_allocator<T,N> >::
tensor_memory( [[maybe_unused]] const Alloc& alloc ) noexcept :
  alloc_()
{
}

template < class T, ::std::size_t N >
template < class Alloc, class MappingType >
constexpr tensor_memory< T, util::fixed_size_allocator<T,N> >::
tensor_memory( [[maybe_unused]] const Alloc& alloc, [[maybe_unused]] const MappingType& mapping ) noexcept :
  alloc_()
{
}

template < class T, ::std::size_t N >
constexpr tensor_memory< T, util::fixed_size_allocator<T,N> >&
tensor_memory< T, util::fixed_size_allocator<T,N> >::operator = ( [[maybe_unused]] tensor_memory&& tm ) noexcept
{
  return *this;
}

template < class T, ::std::size_t N >
template < class U, class Alloc >
constexpr void tensor_memory< T, util::fixed_size_allocator<T,N> >::
assign_allocator( [[maybe_unused]] const tensor_memory<U,Alloc>& tm ) noexcept
{
}

//- Data access

// Const data pointer
template < class T, ::std::size_t N >
[[nodiscard]] constexpr typename tensor_memory< T, util::fixed_size_allocator<T,N> >::const_pointer
tensor_memory< T, util::fixed_size_allocator<T,N> >::data() const noexcept
{
  return this->alloc_.data();
}

// Data pointer
template < class T, ::std::size_t N >
[[nodiscard]] constexpr typename tensor_memory< T, util::fixed_size_allocator<T,N> >::pointer
tensor_memory< T, util::fixed_size_allocator<T,N> >::data() noexcept
{
  return this->alloc_.data();
}

// Copy of allocator
template < class T, ::std::size_t N >
[[nodiscard]] constexpr typename tensor_memory< T, util::fixed_size_allocator<T,N> >::allocator_type
tensor_memory< T, util::fixed_size_allocator<T,N> >::get_allocator() const noexcept
{
  return allocator_type();
}

//- Memory functions

template < class T, ::std::size_t N >
template < class MappingType >
constexpr void tensor_memory< T, util::fixed_size_allocator<T,N> >::
allocate( [[maybe_unused]] const MappingType& mapping )
{
}

template < class T, ::std::size_t N >
template < class MappingType >
constexpr void tensor_memory< T, util::fixed_size_allocator<T,N> >::
deallocate( [[maybe_unused]] const MappingType& mapping )
{
}

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_MEMORY_HPP
