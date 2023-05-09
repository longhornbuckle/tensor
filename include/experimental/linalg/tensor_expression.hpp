//==================================================================================================
//  File:       tensor_expression.hpp
//
//  Summary:    This header defines a tensor expressions.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{

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

// Layout result defines the resultant layout of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct layout_result;

template < readable_tensor Tensor >
struct layout_result< negate_tensor_expression< Tensor > >
{
  using type = typename Tensor::layout_type;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< negate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < readable_tensor Tensor >
struct layout_result< transpose_tensor_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_stride > )
{
  using type = default_layout;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< transpose_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < readable_tensor Tensor >
struct layout_result< conjugate_tensor_expression< Tensor > >
  requires ( ::std::is_same_v< typename Tensor::layout_type, layout_right > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_left > ||
             ::std::is_same_v< typename Tensor::layout_type, layout_stride > )
{
  using type = default_layout;
};

template < unevaluated_tensor_expression Tensor >
struct layout_result< conjugate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};

template < tensor_expression Tensor >
using layout_result_t = typename layout_result< Tensor >::type;


// Accessor result defines the resultant accessor of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct accessor_result;

template < readable_tensor Tensor >
struct accessor_result< negate_tensor_expression< Tensor > >
{
  using type = typename Tensor::accessor_type;
};

template < unevaluated_tensor_expression Tensor >
struct accessor_result< negate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < readable_tensor Tensor >
struct accessor_result< transpose_tensor_expression< Tensor > >
{
  using type = typename Tensor::accessor_type;
};

template < unevaluated_tensor_expression Tensor >
struct accessor_result< conjugate_tensor_expression< Tensor > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

template < class Tensor >
using accessor_result_t = typename accessor_result< Tensor >::type;


// Allocator result defines the resultant allocator of an unevaluated tensor expression.
// It must be defined for any given unevaluated tensor expression.
template < class Tensor >
struct allocator_result
{
  using type = ::std::allocator< typename Tensor::value_type >;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& ) noexcept { return type(); }
};

template < dynamic_tensor Tensor >
struct allocator_result< Tensor >
{
  using type = typename Tensor::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& t ) noexcept { return t.get_allocator(); }
};

template < unary_tensor_expression Tensor >
struct allocator_result< Tensor >
{
  using type = typename allocator_result< decltype( ::std::declval<Tensor>.underlying() ) >::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& t ) noexcept { return allocator_result< decltype( ::std::declval<Tensor>.underlying() ) >::get_allocator( t.underlying(); ) }
};

template < binary_tensor_expression Tensor >
struct allocator_result< Tensor >
{
  using type = typename allocator_result< ::std::conditional_t< dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.first() ) > > ||
                                                                  !dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.second() ) > >,
                                                                decltype( ::std::declval<Tensor>.first() ),
                                                                decltype( ::std::declval<Tensor>.second() ) >::allocator_type;
  [[nodiscard]] static inline constexpr type get_allocator( Tensor&& t ) noexcept
  {
    if constexpr ( dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.first() ) > > ||
                   !dynamic_tensor< ::std_remove_cv_t< decltype( ::std::declval<Tensor>.second() ) > )
    {
      return allocator_result< decltype( ::std::declval<Tensor>.first() ) >::get_allocator( t.first() );
    }
    else
    {
      return allocator_result< decltype( ::std::declval<Tensor>.second() ) >::get_allocator( t.second() );
    }
  }
};

template < class Tensor >
using allocator_result_t = typename allocator_result< Tensor >::type;

// Negate tensor expression
template < tensor_expression Tensor >
class negate_tensor_expression
{
  public:
    // Special member functions
    constexpr negate_tensor_expression( Tensor& t ) noexcept : t_t(t) { }
    constexpr negate_tensor_expression& operator = ( const negate_tensor_expression& t ) noexcept { this->t_ = t.t_; }
    constexpr negate_tensor_expression& operator = ( negate_tensor_expression&& t ) noexcept { this->t_ = t.t_; }
    // Aliases
    using value_type   = decltype( - ::std::declval<typename Tensor::value_type>() );
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename Tensor::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return t_.extents(); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return t_.extent(n); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Negate
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( *this, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( *this, indices ... ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto()
    {
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        // TBD
      }
      else
      {
        return tensor< value_type,
                       extents_type,
                       layout_result_t< Tensor >,
                       typename ::std::allocator_traits< allocator_result_t< Tensor > >::template rebind_t< value_type >,
                       accessor_result_t< Tensor > >
          ( *this, allocator_result< Tensor >::get_allocator( *this ) );
      }
    };

  private:
    // Data
    Tensor& t_;
};

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_HPP
