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
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        // TBD
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
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        // TBD
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



}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_BINARY_TENSOR_EXPRESSIONS_HPP
