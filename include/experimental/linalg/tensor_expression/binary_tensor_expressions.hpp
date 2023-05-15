//==================================================================================================
//  File:       binary_tensor_expressions.hpp
//
//  Summary:    This header defines a binary tensor expressions:
//              add_tensor_expression< Tensor >
//              subtract_tensor_expression< Tensor >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_TENSOR_EXPRESSIONS_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_TENSOR_EXPRESSIONS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{

// Addition tensor expression
template < tensor_expression FirstTensor, tensor_expression SecondTensor >
class add_tensor_expression
  requires ( ( FirstTensor::rank() == SecondTensor::rank() ) &&
             LINALG_DETAIL::extents_maybe_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > )
{
  private:
    using self_type = add_tensor_expression< FirstTensor, SecondTensor >;
  public:
    // Special member functions
    constexpr add_tensor_expression( FirstTensor&& t1, SecondTensor&& t2 )
      noexcept( LINALG_DETAIL::extents_are_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > ) : t1_(t1), t2_(t2)
    {
      if constexpr ( !LINALG_DETAIL::extents_are_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > )
      {
        if ( t1.extents() != t2.extents() )
        {
          throw length_error( "Tensor extents are incompatable." );
        }
      }
    };
    constexpr add_tensor_expression& operator = ( const add_tensor_expression& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
    constexpr add_tensor_expression& operator = ( add_tensor_expression&& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
    // Aliases
    using value_type   = decltype( ::std::declval<typename Tensor::value_type>() + ::std::declval<typename Tensor::value_type>() );
    using index_type   = typename FirstTensor::index_type;
    using size_type    = typename FirstTensor::size_type;
    using extents_type = ::std::conditional_t< ( FirstTensor::extents_type::dynamic_rank() == 0 ) || ( SecondTensor::extents_type::dynamic_rank() != 0 ), typename FirstTensor::extents_type, typename SecondTensor::extents_type >;
    using rank_type    = typename FirstTensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return FirstTensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { if constexpr ( ( FirstTensor::extents_type::dynamic_rank() == 0 ) || ( SecondTensor::extents_type::dynamic_rank() != 0 ) ) { return this->t1_.extents(); } else { return this->t2_.extents(); } }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { if constexpr ( ( FirstTensor::extents_type::dynamic_rank() == 0 ) || ( SecondTensor::extents_type::dynamic_rank() != 0 ) ) { return this->t1_.extent(n); } else { return this->t2_.extent(n); } }
    // Binary tensor expression function
    [[nodiscard]] constexpr const FirstTensor& first() const noexcept { return this->t1_; }
    [[nodiscard]] constexpr const SecondTensor& second() const noexcept { return this->t2_; }
    // Addition
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) + LINALG_DETAIL::access( this->t2_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) + LINALG_DETAIL::access( this->t2_, indices ... ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto()
    {
      // TODO: Optimizations using tensor traits
      // If add and subtracts with the same layout traits can be serialized, then they should.
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          accessor_result_t< self_type > >
          ( *this );
      }
      else
      {
        return tensor< value_type,
                       extents_type,
                       layout_result_t< self_type >,
                       extents_type,
                       typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                       accessor_result_t< self_type > >
          ( *this, allocator_result< self_type >::get_allocator( *this ) );
      }
    };

  private:
    // Data
    FirstTensor&  t1_;
    SecondTensor& t2_;
};

// Subtraction tensor expression
template < tensor_expression FirstTensor, tensor_expression SecondTensor >
class subtract_tensor_expression
  requires ( ( FirstTensor::rank() == SecondTensor::rank() ) &&
             LINALG_DETAIL::extents_maybe_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > )
{
  private:
    using self_type = subtract_tensor_expression< FirstTensor, SecondTensor >;
  public:
    // Special member functions
    constexpr subtract_tensor_expression( FirstTensor&& t1, SecondTensor&& t2 )
      noexcept( LINALG_DETAIL::extents_are_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > ) : t1_(t1), t2_(t2)
    {
      if constexpr ( !LINALG_DETAIL::extents_are_equal_v< typename FirstTensor::extents_type, typename SecondTensor::extents_type > )
      {
        if ( t1.extents() != t2.extents() )
        {
          throw length_error( "Tensor extents are incompatable." );
        }
      }
    };
    constexpr subtract_tensor_expression& operator = ( const subtract_tensor_expression& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
    constexpr subtract_tensor_expression& operator = ( subtract_tensor_expression&& t ) noexcept { this->t1_ = t.t1_; this->t2_ = t.t2_; }
    // Aliases
    using value_type   = decltype( ::std::declval<typename Tensor::value_type>() - ::std::declval<typename Tensor::value_type>() );
    using index_type   = typename FirstTensor::index_type;
    using size_type    = typename FirstTensor::size_type;
    using extents_type = ::std::conditional_t< ( FirstTensor::extents_type::dynamic_rank() == 0 ) || ( SecondTensor::extents_type::dynamic_rank() != 0 ), typename FirstTensor::extents_type, typename SecondTensor::extents_type >;
    using rank_type    = typename FirstTensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return FirstTensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { if constexpr ( ( FirstTensor::extents_type::dynamic_rank() == 0 ) || ( SecondTensor::extents_type::dynamic_rank() != 0 ) ) { return this->t1_.extents(); } else { return this->t2_.extents(); } }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { if constexpr ( ( FirstTensor::extents_type::dynamic_rank() == 0 ) || ( SecondTensor::extents_type::dynamic_rank() != 0 ) ) { return this->t1_.extent(n); } else { return this->t2_.extent(n); } }
    // Binary tensor expression function
    [[nodiscard]] constexpr const FirstTensor& first() const noexcept { return this->t1_; }
    [[nodiscard]] constexpr const SecondTensor& second() const noexcept { return this->t2_; }
    // Subtraction
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) - LINALG_DETAIL::access( this->t2_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) - LINALG_DETAIL::access( this->t2_, indices ... ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto()
    {
      // TODO: Optimizations using tensor traits
      // If add and subtracts with the same layout traits can be serialized, then they should.
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          accessor_result_t< self_type > >
          ( *this );
      }
      else
      {
        return tensor< value_type,
                       extents_type,
                       layout_result_t< self_type >,
                       extents_type,
                       typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                       accessor_result_t< self_type > >
          ( *this, allocator_result< self_type >::get_allocator( *this ) );
      }
    };
  private:
    // Data
    FirstTensor&  t1_;
    SecondTensor& t2_;
};

template < class S, tensor_expression Tensor >
class scalar_preprod_tensor_expression requires requires ( const S& s, const typename Tensor::value_type& v ) { { s * v; } }
{
  private:
    // Alias for self
    using self_type = scalar_preprod_tensor_expression< T, Tensor >;
  public:
    // Special member functions
    constexpr scalar_preprod_tensor_expression( S&& s, Tensor&& t ) : s_(s), t_(t) { };
    constexpr scalar_preprod_tensor_expression& operator = ( const scalar_preprod_tensor_expression& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
    constexpr scalar_preprod_tensor_expression& operator = ( scalar_preprod_tensor_expression&& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
    // Aliases
    using value_type   = decltype( ::std::declval<S>() + ::std::declval<typename Tensor::value_type>() );
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename Tensor::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extent(n); }
    // Binary tensor expression function
    [[nodiscard]] constexpr const S& first() const noexcept { return this->s_; }
    [[nodiscard]] constexpr const Tensor& second() const noexcept { return this->t_; }
    // Pre-Scalar Multiply
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return this->s_ * LINALG_DETAIL::access( this->t1_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return this->s_ * LINALG_DETAIL::access( this->t1_, indices ... ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto()
    {
      // TODO: Optimizations using tensor traits
      // If add and subtracts with the same layout traits can be serialized, then they should.
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          accessor_result_t< self_type > >
          ( *this );
      }
      else
      {
        return tensor< value_type,
                       extents_type,
                       layout_result_t< self_type >,
                       extents_type,
                       typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                       accessor_result_t< self_type > >
          ( *this, allocator_result< self_type >::get_allocator( *this ) );
      }
    };
  private:
    // Data
    S&      s_;
    Tensor& t_;
};

template < class S, tensor_expression Tensor >
class scalar_postprod_tensor_expression requires requires ( const S& s, const typename Tensor::value_type& v ) { { v * s; } }
{
  private:
    // Alias for self
    using self_type = scalar_postprod_tensor_expression< T, Tensor >;
  public:
    // Special member functions
    constexpr scalar_postprod_tensor_expression( Tensor&& t, S&& s ) : t_(t), s_(s) { };
    constexpr scalar_postprod_tensor_expression& operator = ( const scalar_postprod_tensor_expression& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
    constexpr scalar_postprod_tensor_expression& operator = ( scalar_postprod_tensor_expression&& t ) noexcept { this->t_ = t.t_; this->s_ = t.s_; }
    // Aliases
    using value_type   = decltype( ::std::declval<S>() + ::std::declval<typename Tensor::value_type>() );
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename Tensor::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extent(n); }
    // Binary tensor expression function
    [[nodiscard]] constexpr const S& first() const noexcept { return this->s_; }
    [[nodiscard]] constexpr const Tensor& second() const noexcept { return this->t_; }
    // Pre-Scalar Multiply
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) * this->s_; }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( *this, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t1_, indices ... ) * this->s_; }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto()
    {
      // TODO: Optimizations using tensor traits
      // If add and subtracts with the same layout traits can be serialized, then they should.
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          accessor_result_t< self_type > >
          ( *this );
      }
      else
      {
        return tensor< value_type,
                       extents_type,
                       layout_result_t< self_type >,
                       extents_type,
                       typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                       accessor_result_t< self_type > >
          ( *this, allocator_result< self_type >::get_allocator( *this ) );
      }
    };
  private:
    // Data
    Tensor& t_;
    S&      s_;
};

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_TENSOR_EXPRESSIONS_HPP
