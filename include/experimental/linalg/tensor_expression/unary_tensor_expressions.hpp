//==================================================================================================
//  File:       unary_tensor_expressions.hpp
//
//  Summary:    This header defines a unary tensor expressions:
//              negate_tensor_expression< Tensor >
//              transpose_tensor_expression< Tensor >
//              conjugate_tensor_expression< Tensor >
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{

// Negation tensor expression
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor >
#else
template < class Tensor, typename >
#endif
class negate_tensor_expression
{
  private:
    using self_type = negate_tensor_expression< Tensor >;
  public:
    // Special member functions
    constexpr negate_tensor_expression( Tensor&& t ) noexcept : t_(t) { }
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
    [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::dynamic_rank() == 0 ) ?
                                                      ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                                                                    extents_type,
                                                                                                    layout_result_t< self_type >,
                                                                                                    accessor_result_t< self_type > >,
                                                                                         self_type > :
                                                      ::std::is_nothrow_constructible_v< tensor< value_type,
                                                                                                 extents_type,
                                                                                                 layout_result_t< self_type >,
                                                                                                 extents_type,
                                                                                                 typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                                                                                                 accessor_result_t< self_type > >,
                                                                                         self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
    {
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
};

// Transpose
template < ::std::size_t index1, ::std::size_t index2 >
struct transpose_indices_t
{
  [[nodiscard]] constexpr ::std::size_t first() noexcept { return index1; }
  [[nodiscard]] constexpr ::std::size_t second() noexcept { return index2; }
};

struct transpose_indices_v
{
  constexpr transpose_indices_v() noexcept : index1(0), index2(1) {}
  constexpr transpose_indices_v( const transpose_indices_v& ) noexcept = default;
  constexpr transpose_indices_v( transpose_indices_v&& ) noexcept = default;
  [[nodiscard]] constexpr ::std::size_t first() noexcept { return this->index1; }
  [[nodiscard]] constexpr ::std::size_t second() noexcept { return this->index2; }
private:
  ::std::size_t index1;
  ::std::size_t index2;
};

template < class Extents, class TransposeIndices >
class transpose_helper;
template < class Extents >
class transpose_helper< Extents, transpose_indices_v >
{
  private:
    template < class I >
    struct helper;
    template < class R, auto ... Indices >
    struct helper< ::std::integer_sequence< R, Indices ... > >
    {
      [[nodiscard]] constexpr auto index( auto i, const transpose_indices_v& transpose_indices ) noexcept
      {
        if ( i == transpose_indices.first() )
        {
          return transpose_indices.second();
        }
        else if ( i == transpose_indices.second() )
        {
          return transpose_indices.first();
        }
        else
        {
          return i;
        }
      }
      [[nodiscard]] constexpr extents_type extents( const Extents& e, const transpose_indices_v& transpose_indices ) noexcept
      {
        return extents_type( e.extent( index( Indices, transpose_indices ) ) ... );
      }
    };
  public:
    using extents_type = ::std::conditional_t< extent_helper< Extents >::all_extents_equal_v,
                                               Extents,
                                               ::std::experimental::dextents< typename Extents::size_type, Extents::rank() > >;
    [[nodiscard]] constexpr extents_type extents( const Extents& e, const transpose_indices_v& transpose_indices ) noexcept
    {
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return extents_type { };
      }
      else
      {
        return helper< extents_type, ::std::integer_sequence< typename extents_type::rank_type, Extents::rank() > >::extents( e, transpose_indices );
      }
    }
    [[nodiscard]] constexpr extents_type extent( const Extents& e, const transpose_indices_v& transpose_indices, const extents_type::rank_type n ) noexcept
    {
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return extents_type::static_extent( n );
      }
      else
      {
        if ( n == transpose_indices().first() )
        {
          return e.extent( transpose_indices.second() );
        }
        else if ( n == transpose_indices().second() )
        {
          return e.extent( transpose_indices.first() )
        }
        else
        {
          return e.extent( n );
        }
      }
    }
    template < class Tensor >
    [[nodiscard]] constexpr auto access( Tensor&& t, const transpose_indices_v& transpose_indices, auto ... indices ) noexcept
    {
      return LINALG_DETAIL::access( t, index( indices, transpose_indices ) ... );
    }
};
template < ::std::size_t index1, ::std::size_t index2 >
class transpose_helper< transpose_indices_t<index1,index2> >
{
  private:
    template < class I >
    struct helper;
    template < class R, auto ... Indices >
    struct helper< ::std::integer_sequence< R, Indices ... > >
    {
      [[nodiscard]] constexpr auto index( auto i ) noexcept
      {
        if ( i == index1 )
        {
          return index2;
        }
        else if ( i == index2 )
        {
          return index1;
        }
        else
        {
          return i;
        }
      }
      using extents_type = ::std::experimental::extents< typename Extents::size_type, Extents::static_extent( index( Indices ) ) ... >;
      [[nodiscard]] constexpr extents_type extents( const Extents& e, const transpose_indices_v& transpose_indices ) noexcept
      {
        return extents_type( e.extent( index( Indices ) ) ... );
      }
    };
  public:
    using extents_type = helper< extents_type, ::std::integer_sequence< typename extents_type::rank_type, Extents::rank() > >::extents_type;
    [[nodiscard]] constexpr extents_type extents( const Extents& e, const transpose_indices_v& transpose_indices ) noexcept
    {
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return extents_type { };
      }
      else
      {
        return helper< extents_type, ::std::integer_sequence< typename extents_type::rank_type, Extents::rank() > >::extents( e, transpose_indices );
      }
    }
    [[nodiscard]] constexpr extents_type extent( const Extents& e, const transpose_indices_v& transpose_indices, const extents_type::rank_type n ) noexcept
    {
      if constexpr ( extents_type::dynamic_rank() == 0 )
      {
        return extents_type::static_extent( n );
      }
      else
      {
        if ( n == transpose_indices().first() )
        {
          return e.extent( transpose_indices.second() );
        }
        else if ( n == transpose_indices().second() )
        {
          return e.extent( transpose_indices.first() )
        }
        else
        {
          return e.extent( n );
        }
      }
    }
    template < class Tensor >
    [[nodiscard]] constexpr auto access( Tensor&& t, const transpose_indices_v& transpose_indices, auto ... indices ) noexcept
    {
      return LINALG_DETAIL::access( t, index( indices ) ... );
    }
};

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor, class Transpose >
#else
template < class Tensor, class Transpose, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > > >
#endif
class transpose_tensor_expression
  requires ( Tensor::rank() > 1 )
{
  private:
    using self_type = transpose_tensor_expression< Tensor, Transpose >;
  public:
    // Special member functions
    constexpr transpose_tensor_expression( Tensor&& t, Transpose&& indices = Transpose() ) noexcept : t_(t), indices_(indices) { }
    constexpr transpose_tensor_expression& operator = ( const transpose_tensor_expression& t ) noexcept { this->t_ = t.t_; this->indices_ = t.indices_; }
    constexpr transpose_tensor_expression& operator = ( transpose_tensor_expression&& t ) noexcept { this->t_ = t.t_;; this->indices_ = t.indices_; }
    // Aliases
    using value_type   = typename Tensor::value_type;
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename transpose_helper< typename Tensor::extents_type, Transpose >::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return transpose_helper< typename Tensor::extents_type, Transpose >::extents( this->t_.extents() ); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return transpose_helper< typename Tensor::extents_type, Transpose >::extent( this->t_.extents(), n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Transpose
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::dynamic_rank() == 0 ) ?
                                                      ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                                                                    extents_type,
                                                                                                    layout_result_t< self_type >,
                                                                                                    accessor_result_t< self_type > >,
                                                                                         self_type > :
                                                      ::std::is_nothrow_constructible_v< tensor< value_type,
                                                                                                 extents_type,
                                                                                                 layout_result_t< self_type >,
                                                                                                 extents_type,
                                                                                                 typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                                                                                                 accessor_result_t< self_type > >,
                                                                                         self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
    {
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
  Tensor                                    t_;
  [[no_unique_address]] transpose_indices_t indices_;
};

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::vector_expression Tensor >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< Tensor > > >
#endif
class transpose_tensor_expression
{
  private:
    using self_type = transpose_tensor_expression< Tensor >;
  public:
    // Special member functions
    constexpr transpose_tensor_expression( Tensor&& t ) noexcept : t_(t) { }
    constexpr transpose_tensor_expression& operator = ( const transpose_tensor_expression& t ) noexcept { this->t_ = t.t_; }
    constexpr transpose_tensor_expression& operator = ( transpose_tensor_expression&& t ) noexcept { this->t_ = t.t_; }
    // Aliases
    using value_type   = typename Tensor::value_type;
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename Tensor::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extents( n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Transpose
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( this->t_, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( LINALG_DETAIL::access( this->t_, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return LINALG_DETAIL::access( this->t_, indices ... ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::dynamic_rank() == 0 ) ?
                                                      ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                                                                    extents_type,
                                                                                                    layout_result_t< self_type >,
                                                                                                    accessor_result_t< self_type > >,
                                                                                         self_type > :
                                                      ::std::is_nothrow_constructible_v< tensor< value_type,
                                                                                                 extents_type,
                                                                                                 layout_result_t< self_type >,
                                                                                                 extents_type,
                                                                                                 typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                                                                                                 accessor_result_t< self_type > >,
                                                                                         self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
    {
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
  Tensor t_;
};

// Conjugate
#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::tensor_expression Tensor, class Transpose >
#else
template < class Tensor, class Transpose, typename = ::std::enable_if_t< LINALG_CONCEPTS::tensor_expression_v< Tensor > > >
#endif
class conjugate_tensor_expression
  requires ( Tensor::rank() > 1 )
{
  private:
    using self_type = conjugate_tensor_expression< Tensor, Transpose >;
  public:
    // Special member functions
    constexpr conjugate_tensor_expression( Tensor&& t, Transpose&& indices = Transpose() ) noexcept : t_(t), indices_(indices) { }
    constexpr conjugate_tensor_expression& operator = ( const conjugate_tensor_expression& t ) noexcept { this->t_ = t.t_; this->indices_ = t.indices_; }
    constexpr conjugate_tensor_expression& operator = ( conjugate_tensor_expression&& t ) noexcept { this->t_ = t.t_;; this->indices_ = t.indices_; }
    // Aliases
    using value_type   = typename Tensor::value_type;
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename transpose_helper< typename Tensor::extents_type, Transpose >::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return transpose_helper< typename Tensor::extents_type, Transpose >::extents( this->t_.extents() ); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return transpose_helper< typename Tensor::extents_type, Transpose >::extent( this->t_.extents(), n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Conjugate
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( ::std::conj( transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ) ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return ::std::conj( transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ) ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( ::std::conj( transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ) ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return ::std::conj( transpose_helper< typename Tensor::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ) ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::dynamic_rank() == 0 ) ?
                                                      ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                                                                    extents_type,
                                                                                                    layout_result_t< self_type >,
                                                                                                    accessor_result_t< self_type > >,
                                                                                         self_type > :
                                                      ::std::is_nothrow_constructible_v< tensor< value_type,
                                                                                                 extents_type,
                                                                                                 layout_result_t< self_type >,
                                                                                                 extents_type,
                                                                                                 typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                                                                                                 accessor_result_t< self_type > >,
                                                                                         self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
    {
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
  Tensor                                    t_;
  [[no_unique_address]] transpose_indices_t indices_;
};

#ifdef LINALG_ENABLE_CONCEPTS
template < LINALG_CONCEPTS::vector_expression Tensor >
#else
template < class Tensor, typename = ::std::enable_if_t< LINALG_CONCEPTS::vector_expression_v< Tensor > > >
#endif
class conjugate_tensor_expression
{
  private:
    using self_type = conjugate_tensor_expression< Tensor, Transpose >;
  public:
    // Special member functions
    constexpr conjugate_tensor_expression( Tensor&& t ) noexcept : t_(t) { }
    constexpr conjugate_tensor_expression& operator = ( const conjugate_tensor_expression& t ) noexcept { this->t_ = t.t_; }
    constexpr conjugate_tensor_expression& operator = ( conjugate_tensor_expression&& t ) noexcept { this->t_ = t.t_; }
    // Aliases
    using value_type   = typename Tensor::value_type;
    using index_type   = typename Tensor::index_type;
    using size_type    = typename Tensor::size_type;
    using extents_type = typename Tensor::extents_type;
    using rank_type    = typename Tensor::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return Tensor::rank(); }
    [[nodiscard]] constexpr extents_type extents() noexcept { return this->t_.extents(); }
    [[nodiscard]] constexpr size_type extents( rank_type n ) noexcept { return this->t_.extents( n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Conjugate
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( ::std::conj( LINALG_DETAIL::access( this->t_, indices ... ) ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return ::std::conj( LINALG_DETAIL::access( this->t_, indices ... ) ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr const_reference operator()( IndexType ... indices ) const noexcept( noexcept( ::std::conj( LINALG_DETAIL::access( this->t_, indices ... ) ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return ::std::conj( LINALG_DETAIL::access( this->t_, indices ... ) ); }
    #endif
    // Implicit conversion
    [[nodiscard]] constexpr operator auto() noexcept( ( extents_type::dynamic_rank() == 0 ) ?
                                                      ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                                                                    extents_type,
                                                                                                    layout_result_t< self_type >,
                                                                                                    accessor_result_t< self_type > >,
                                                                                         self_type > :
                                                      ::std::is_nothrow_constructible_v< tensor< value_type,
                                                                                                 extents_type,
                                                                                                 layout_result_t< self_type >,
                                                                                                 extents_type,
                                                                                                 typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_t< value_type >,
                                                                                                 accessor_result_t< self_type > >,
                                                                                         self_type, decltype( allocator_result< self_type >::get_allocator( ::std::declval< self_type >() ) ) > )
    {
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
  Tensor t_;
};

}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP
