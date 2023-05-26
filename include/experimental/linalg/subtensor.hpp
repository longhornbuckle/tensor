//==================================================================================================
//  File:       subtensor.hpp
//
//  Summary:    This header defines a subtensor function for getting views into a potentially larger
//              tensor construct.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_SUBTENSOR_HPP
#define LINEAR_ALGEBRA_SUBTENSOR_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

template < class T,
           class Extents,
           class LayoutPolicy,
           class CapExtents,
           class Allocator,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subvector( dr_tensor< T, Extents, LayoutPolicy, CapExtents, Allocator, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 1 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class CapExtents,
           class Allocator,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subvector( const dr_tensor< T, Extents, LayoutPolicy, CapExtents, Allocator, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 1 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class CapExtents,
           class Allocator,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto submatrix( dr_tensor< T, Extents, LayoutPolicy, CapExtents, Allocator, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 2 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class CapExtents,
           class Allocator,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto submatrix( const dr_tensor< T, Extents, LayoutPolicy, CapExtents, Allocator, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 2 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class CapExtents,
           class Allocator,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subtensor( const dr_tensor< T, Extents, LayoutPolicy, CapExtents, Allocator, AccessorPolicy >& t, SliceArgs&& ... args )
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class CapExtents,
           class Allocator,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subtensor( dr_tensor< T, Extents, LayoutPolicy, CapExtents, Allocator, AccessorPolicy >& t, SliceArgs&& ... args )
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subvector( fs_tensor< T, Extents, LayoutPolicy, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 1 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subvector( const fs_tensor< T, Extents, LayoutPolicy, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 1 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto submatrix( fs_tensor< T, Extents, LayoutPolicy, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 2 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto submatrix( const fs_tensor< T, Extents, LayoutPolicy, AccessorPolicy >& t, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                                       args ... ) )::rank() == 2 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subtensor( fs_tensor< T, Extents, LayoutPolicy, AccessorPolicy >& t, SliceArgs&& ... args )
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class T,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subtensor( const fs_tensor< T, Extents, LayoutPolicy, AccessorPolicy >& t, SliceArgs&& ... args )
  noexcept( noexcept( ::std::experimental::submdspan( ::std::mdspan< ::std::remove_pointer_t< decltype( t.data_handle() ) >,
                                                                     Extents,
                                                                     LayoutPolicy,
                                                                     ::std::remove_reference_t< decltype( t.accessor() ) > >( t.data_handle(), t.mapping(), t.accessor() ),
                                                      args ... ) ) )
{
  return ::std::experimental::submdspan( ::std::mdspan( t.data_handle(), t.mapping(), t.accessor() ),
                                         args ... );
}

template < class ElementType,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subvector( ::std::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >& view, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( view, args ... ) )::rank() == 1 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( view, args ... ) ) )
{
  return ::std::experimental::submdspan( view, args ... );
}

template < class ElementType,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subvector( const ::std::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >& view, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( view, args ... ) )::rank() == 1 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( view, args ... ) ) )
{
  return ::std::experimental::submdspan( view, args ... );
}

template < class ElementType,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto submatrix( ::std::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >& view, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( view, args ... ) )::rank() == 2 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( view, args ... ) ) )
{
  return ::std::experimental::submdspan( view, args ... );
}

template < class ElementType,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto submatrix( const ::std::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >& view, SliceArgs&& ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( view, args ... ) )::rank() == 2 )
#endif
  noexcept( noexcept( ::std::experimental::submdspan( view, args ... ) ) )
{
  return ::std::experimental::submdspan( view, args ... );
}

template < class ElementType,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subtensor( ::std::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >& view, SliceArgs&& ... args )
  noexcept( noexcept( ::std::experimental::submdspan( view, args ... ) ) )
{
  return ::std::experimental::submdspan( view, args ... );
}

template < class ElementType,
           class Extents,
           class LayoutPolicy,
           class AccessorPolicy,
           class ... SliceArgs >
[[nodiscard]] constexpr auto subtensor( const ::std::mdspan< ElementType, Extents, LayoutPolicy, AccessorPolicy >& view, SliceArgs&& ... args )
  noexcept( noexcept( ::std::experimental::submdspan( view, args ... ) ) )
{
  return ::std::experimental::submdspan( view, args ... );
}

LINALG_END // end linalg namespace

#endif  //- LINEAR_ALGEBRA_SUBTENSOR_HPP
