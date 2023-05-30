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

LINALG_EXPRESSIONS_BEGIN // expressions namespace

template < class Tensor, class Traits >
class unary_tensor_expression_base;

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class > class UTE,
                              class Tensor,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor >, Traits >
#else
template < template < class, class > class UTE,
                                     class Tensor,
                                     typename Enable,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor, Enable >, Traits >
#endif
{
  private :
    #ifdef LINALG_ENABLE_CONCEPTS
    using expression_type = UTE< Tensor >;
    #else
    using expression_type = UTE< Tensor, Enable >;
    #endif
    using traits_type     = Traits;
  public:
    // Aliases
    using value_type   = typename traits_type::value_type;
    using index_type   = typename traits_type::index_type;
    using size_type    = typename traits_type::size_type;
    using extents_type = typename traits_type::extents_type;
    using rank_type    = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return expression_type::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return static_cast< const expression_type* >(this)->extents(); }
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept { return static_cast< const expression_type* >(this)->extent( n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr decltype(auto) underlying() const noexcept { return static_cast< const expression_type* >(this)->underlying(); }
    // Index access
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator[]( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator[]( indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator()( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator()( indices ... ); }
    #endif
};

#ifdef LINALG_ENABLE_CONCEPTS
template < template < class, class > class UTE,
                                     class Tensor,
                                     class Param,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor, Param >, Traits >
#else
template < template < class, class, class > class UTE,
                                            class Tensor,
                                            class Param,
                                            typename Enable,
           class Traits >
class unary_tensor_expression_base< UTE< Tensor, Param, Enable >, Traits >
#endif
{
  private :
    #ifdef LINALG_ENABLE_CONCEPTS
    using expression_type = UTE< Tensor, Param >;
    #else
    using expression_type = UTE< Tensor, Param, Enable >;
    #endif
    using traits_type     = Traits;
  public:
    // Aliases
    using value_type   = typename traits_type::value_type;
    using index_type   = typename traits_type::index_type;
    using size_type    = typename traits_type::size_type;
    using extents_type = typename traits_type::extents_type;
    using rank_type    = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return expression_type::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return static_cast< const expression_type* >(this)->extents(); }
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept { return static_cast< const expression_type* >(this)->extent( n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr decltype(auto) underlying() const noexcept { return static_cast< const expression_type* >(this)->underlying(); }
    // Index access
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator[]( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator[]( indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( static_cast< const expression_type* >(this)->operator()( indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return static_cast< const expression_type* >(this)->operator()( indices ... ); }
    #endif
};

template < class Tensor >
class negate_tensor_expression_traits
{
  public:
    using value_type   = ::std::remove_cv_t< decltype( - ::std::declval< typename ::std::remove_reference_t< Tensor >::value_type >() ) >;
    using index_type   = typename ::std::remove_reference_t< Tensor >::index_type;
    using size_type    = typename ::std::remove_reference_t< Tensor >::size_type;
    using extents_type = typename ::std::remove_reference_t< Tensor >::extents_type;
    using rank_type    = typename ::std::remove_reference_t< Tensor >::rank_type;
};

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
class negate_tensor_expression : public unary_tensor_expression_base< negate_tensor_expression< Tensor >, negate_tensor_expression_traits< Tensor > >
#else
template < class Tensor, typename Enable >
class negate_tensor_expression : public unary_tensor_expression_base< negate_tensor_expression< Tensor, Enable >, negate_tensor_expression_traits< Tensor > >
#endif
{
  private:
    // Aliases
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = negate_tensor_expression< Tensor >;
    #else
    using self_type   = negate_tensor_expression< Tensor, Enable >;
    #endif
    using traits_type = negate_tensor_expression_traits< Tensor >;
    using base_type   = unary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr negate_tensor_expression( Tensor&& t ) noexcept : t_(t) { }
    constexpr negate_tensor_expression& operator = ( const negate_tensor_expression& t ) noexcept { this->t_ = t.t_; }
    constexpr negate_tensor_expression& operator = ( negate_tensor_expression&& t ) noexcept { this->t_ = t.t_; }
    // Aliases
    using value_type   = typename traits_type::value_type;
    using index_type   = typename traits_type::index_type;
    using size_type    = typename traits_type::size_type;
    using extents_type = typename traits_type::extents_type;
    using rank_type    = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< Tensor >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return t_.extents(); }
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept { return t_.extent(n); }
    // Unary tensor expression function
    [[nodiscard]] constexpr decltype(auto) underlying() const noexcept { return this->t_; }
    // Negate
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( - LINALG_DETAIL::access( ::std::declval<Tensor&&>(), indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( this->t_, indices ... ); }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( - LINALG_DETAIL::access( ::std::declval<Tensor&&>(), indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
      { return - LINALG_DETAIL::access( this->t_, indices ... ); }
    #endif
  private:
    // Define noexcept specification of conversion operator
    [[nodiscard]] static inline constexpr bool conversion_is_noexcept() noexcept
    {
      if constexpr( extents_type::rank_dynamic() == 0 )
      {
        return ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                             extents_type,
                                                             layout_result_t< self_type >,
                                                             accessor_result_t< self_type > >,
                                                  const base_type& >;
      }
      else
      {
        return ::std::is_nothrow_constructible_v< dr_tensor< value_type,
                                                             extents_type,
                                                             layout_result_t< self_type >,
                                                             extents_type,
                                                             typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_alloc< value_type >,
                                                             accessor_result_t< self_type > >,
                                                  const base_type&,
                                                  decltype( allocator_result< const self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
      }
    }
  public:
    // Implicit conversion
    [[nodiscard]] constexpr operator auto() const noexcept( conversion_is_noexcept() )
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          accessor_result_t< self_type > >
          ( *static_cast< const base_type* >( this ) );
      }
      else
      {
        return dr_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          extents_type,
                          typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_alloc< value_type >,
                          accessor_result_t< self_type > >
          ( *static_cast< const base_type* >( this ), allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( *this ) ) );
      }
    }
  private:
    Tensor&& t_;
};

// Transpose
template < ::std::size_t index1, ::std::size_t index2 >
struct transpose_indices_t
{
  [[nodiscard]] constexpr ::std::size_t first() const noexcept { return index1; }
  [[nodiscard]] constexpr ::std::size_t second() const noexcept { return index2; }
};

template < class IndexType1, class IndexType2 >
struct transpose_indices_v
{
  constexpr transpose_indices_v() noexcept : index1_(0), index2_(1) {}
  constexpr transpose_indices_v( IndexType1 index1, IndexType2 index2 ) noexcept : index1_(index1), index2_(index2) {}
  constexpr transpose_indices_v( const transpose_indices_v& ) noexcept = default;
  constexpr transpose_indices_v( transpose_indices_v&& ) noexcept = default;
  [[nodiscard]] constexpr IndexType1 first() const noexcept { return this->index1_; }
  [[nodiscard]] constexpr IndexType2 second() const noexcept { return this->index2_; }
private:
  IndexType1 index1_;
  IndexType2 index2_;
};

template < class Extents, class TransposeIndices >
class transpose_helper;
template < class Extents, class IndexType1, class IndexType2 >
class transpose_helper< Extents, transpose_indices_v< IndexType1, IndexType2 > >
{
  private:
    template < class >
    struct extents_helper;
    template < class SizeType, ::std::size_t ... Indices >
    struct extents_helper< ::std::extents< SizeType, Indices ... > >
    {
    private:
      [[nodiscard]] static inline constexpr auto val() noexcept { return ::std::get<0>( tuple( Indices ... ) ); }
      [[nodiscard]] static inline constexpr auto dval( [[maybe_unused]] ::std::size_t ) noexcept { return ::std::dynamic_extent; }
    public:
      static inline constexpr bool all_extents_equal_v = ( ( Indices == val() ) && ... );
      using dtype = ::std::extents< SizeType, dval( Indices ) ... >;
    };
    template < class I >
    struct helper;
    template < class R, auto ... Indices >
    struct helper< ::std::integer_sequence< R, Indices ... > >
    {
      template < class OtherIndexType1, class OtherIndexType2, auto Index, class ... OtherIndexType >
      [[nodiscard]] static inline constexpr auto index( const transpose_indices_v< OtherIndexType1, OtherIndexType2 >& transpose_indices, OtherIndexType ... indices ) noexcept
      {
        using array_type = ::std::array< ::std::reference_wrapper< ::std::remove_reference_t< decltype( ::std::get< 0 >( ::std::forward_as_tuple( indices ... ) ) ) > >, sizeof...(indices) >;
        if ( Index == transpose_indices.first() )
        {
          return array_type( { indices ... } )[ transpose_indices.second() ].get();
        }
        else if ( Index == transpose_indices.second() )
        {
          return array_type( { indices ... } )[ transpose_indices.first() ].get();
        }
        else
        {
          return ::std::get< Index >( ::std::forward_as_tuple( indices ... ) );
        }
      }
      template < class Tensor, class OtherIndexType1, class OtherIndexType2, class ... OtherIndexType >
      [[nodiscard]] static inline constexpr decltype(auto) access( Tensor&& t, const transpose_indices_v< OtherIndexType1, OtherIndexType2 >& transpose_indices, OtherIndexType ... indices ) noexcept
      {
        return LINALG_DETAIL::access( t, index< OtherIndexType1, OtherIndexType2, Indices, OtherIndexType ... >( transpose_indices, indices ... ) ... );
      }
      template < class OtherIndexType1, class OtherIndexType2 >
      [[nodiscard]] static inline constexpr Extents extents( const Extents& e, const transpose_indices_v< OtherIndexType1, OtherIndexType2 >& transpose_indices ) noexcept
      {
        return Extents( e.extent( index< OtherIndexType1, OtherIndexType2, Indices, decltype(Indices) ... >( transpose_indices, Indices ... ) ) ... );
      }
    };
  public:
    using extents_type = ::std::conditional_t< extents_helper< Extents >::all_extents_equal_v,
                                               Extents,
                                               typename extents_helper< Extents >::dtype >;
    template < class OtherIndexType1, class OtherIndexType2 >
    [[nodiscard]] static inline constexpr extents_type extents( const Extents& e, const transpose_indices_v< OtherIndexType1, OtherIndexType2 >& transpose_indices ) noexcept
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return extents_type { };
      }
      else
      {
        return helper< ::std::make_integer_sequence< typename extents_type::rank_type, Extents::rank() > >::extents( e, transpose_indices );
      }
    }
    template < class OtherIndexType1, class OtherIndexType2 >
    [[nodiscard]] static inline constexpr auto extent( const Extents& e, const transpose_indices_v< OtherIndexType1, OtherIndexType2 >& transpose_indices, const typename extents_type::rank_type n ) noexcept
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return extents_type::static_extent( n );
      }
      else
      {
        if ( n == transpose_indices.first() )
        {
          return e.extent( transpose_indices.second() );
        }
        else if ( n == transpose_indices.second() )
        {
          return e.extent( transpose_indices.first() );
        }
        else
        {
          return e.extent( n );
        }
      }
    }
    template < class Tensor, class OtherIndexType1, class OtherIndexType2, class ... OtherIndexType >
    [[nodiscard]] static inline constexpr auto access( Tensor&& t, const transpose_indices_v< OtherIndexType1, OtherIndexType2 >& transpose_indices, OtherIndexType ... indices ) noexcept
    {
      return helper< ::std::make_integer_sequence< typename extents_type::rank_type, extents_type::rank() > >::access( t, transpose_indices, indices ... );
    }
};
template < class Extents, ::std::size_t index1, ::std::size_t index2 >
class transpose_helper< Extents, transpose_indices_t<index1,index2> >
{
  private:
    template < class I >
    struct helper;
    template < class R, auto ... Indices >
    struct helper< ::std::integer_sequence< R, Indices ... > >
    {
      template < auto Index, class ... OtherIndexType >
      [[nodiscard]] static inline constexpr auto index( OtherIndexType ... indices ) noexcept
      {
        if constexpr ( Extents::rank() == 1 )
        {
          return ::std::get< Index >( ::std::forward_as_tuple( indices ... ) );
        }
        else
        {
          if constexpr ( Index == index1 )
          {
            return ::std::get< index2 >( ::std::forward_as_tuple( indices ... ) );
          }
          else if constexpr ( Index == index2 )
          {
            return ::std::get< index1 >( ::std::forward_as_tuple( indices ... ) );
          }
          else
          {
            return ::std::get< Index >( ::std::forward_as_tuple( indices ... ) );
          }
        }
      }
      template < class Tensor, class ... OtherIndexType >
      [[nodiscard]] static inline constexpr decltype(auto) access( Tensor&& t, OtherIndexType ... indices ) noexcept
      {
        return LINALG_DETAIL::access( t, index< Indices, OtherIndexType ... >( indices ... ) ... );
      }
      template < auto Rank >
      struct extents_helper
      {
        using type = ::std::extents< typename Extents::size_type, Extents::static_extent( index< Indices >( Indices ... ) ) ... >;
      };
      struct vector_extents_helper
      {
        using type = Extents;
      };
      using extents_type = typename ::std::conditional_t< ( Extents::rank() > 1 ),
                                                          extents_helper< Extents::rank() >,
                                                          vector_extents_helper >::type;
      [[nodiscard]] static inline constexpr extents_type extents( const Extents& e ) noexcept
      {
        return extents_type( e.extent( index< Indices >( Indices ... ) ) ... );
      }
    };
  public:
    using extents_type = typename helper< ::std::make_integer_sequence< typename Extents::rank_type, Extents::rank() > >::extents_type;
    [[nodiscard]] static inline constexpr extents_type extents( const Extents& e, [[maybe_unused]] const transpose_indices_t< index1, index2 >& transpose_indices ) noexcept
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return extents_type { };
      }
      else
      {
        return helper< ::std::make_integer_sequence< typename extents_type::rank_type, Extents::rank() > >::extents( e );
      }
    }
    [[nodiscard]] static inline constexpr auto extent( const Extents& e, [[maybe_unused]] const transpose_indices_t< index1, index2 >& transpose_indices, const typename extents_type::rank_type n ) noexcept
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return extents_type::static_extent( n );
      }
      else
      {
        if ( n == index1 )
        {
          return e.extent( index2 );
        }
        else if ( n == index2 )
        {
          return e.extent( index1 );
        }
        else
        {
          return e.extent( n );
        }
      }
    }
    template < class Tensor, class ... OtherIndexType >
    [[nodiscard]] static inline constexpr decltype(auto) access( Tensor&& t, [[maybe_unused]] const transpose_indices_t< index1, index2 >& transpose_indices, OtherIndexType ... indices ) noexcept
    {
      return helper< ::std::make_integer_sequence< typename extents_type::rank_type, extents_type::rank() > >::access( t, indices ... );
    }
};

template < class Tensor, class Transpose >
class transpose_tensor_expression_traits
{
  public:
    using value_type   = typename ::std::remove_reference_t< Tensor >::value_type;
    using index_type   = typename ::std::remove_reference_t< Tensor >::index_type;
    using size_type    = typename ::std::remove_reference_t< Tensor >::size_type;
    using extents_type = typename transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::extents_type;
    using rank_type    = typename ::std::remove_reference_t< Tensor >::rank_type;
};

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
class transpose_tensor_expression : public unary_tensor_expression_base< transpose_tensor_expression< Tensor, Transpose >, transpose_tensor_expression_traits< Tensor, Transpose > >
#else
template < class Tensor, class Transpose, typename Enable >
class transpose_tensor_expression : public unary_tensor_expression_base< transpose_tensor_expression< Tensor, Transpose, Enable >, transpose_tensor_expression_traits< Tensor, Transpose > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = transpose_tensor_expression< Tensor, Transpose >;
    #else
    using self_type   = transpose_tensor_expression< Tensor, Transpose, Enable >;
    #endif
    using traits_type = transpose_tensor_expression_traits< Tensor, Transpose >;
    using base_type   = unary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr transpose_tensor_expression( Tensor&& t, Transpose&& indices = Transpose() ) noexcept : t_(t), indices_(indices) { }
    constexpr transpose_tensor_expression& operator = ( const transpose_tensor_expression& t ) noexcept { this->t_ = t.t_; this->indices_ = t.indices_; }
    constexpr transpose_tensor_expression& operator = ( transpose_tensor_expression&& t ) noexcept { this->t_ = t.t_;; this->indices_ = t.indices_; }
    // Aliases
    using value_type   = typename traits_type::value_type;
    using index_type   = typename traits_type::index_type;
    using size_type    = typename traits_type::size_type;
    using extents_type = typename traits_type::extents_type;
    using rank_type    = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< Tensor >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::extents( this->t_.extents(), this->indices_ ); }
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept { return transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::extent( this->t_.extents(), this->indices_, n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Transpose
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
    {
      if constexpr ( ::std::remove_reference_t< Tensor >::rank() == 1 )
      {
        return LINALG_DETAIL::access( this->t_, indices ... );
      }
      else
      {
        return transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ... );
      }
    }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(OtherIndexType) == rank() ) && ( ::std::is_convertible_v<OtherIndexType,index_type> && ... )
    #endif
    {
      if constexpr ( ::std::remove_reference_t< Tensor >::rank() == 1 )
      {
        return LINALG_DETAIL::access( this->t_, indices ... );
      }
      else
      {
        return transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ... );
      }
    }
    #endif
  private:
    // Define noexcept specification of conversion operator
    [[nodiscard]] static inline constexpr bool conversion_is_noexcept() noexcept
    {
      if constexpr( extents_type::rank_dynamic() == 0 )
      {
        return ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                             extents_type,
                                                             layout_result_t< self_type >,
                                                             accessor_result_t< self_type > >,
                                                  const base_type& >;
      }
      else
      {
        return ::std::is_nothrow_constructible_v< dr_tensor< value_type,
                                                             extents_type,
                                                             layout_result_t< self_type >,
                                                             extents_type,
                                                             typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_alloc< value_type >,
                                                             accessor_result_t< self_type > >,
                                                  const base_type&,
                                                  decltype( allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
      }
    }
  public:
    // Implicit conversion
    [[nodiscard]] constexpr operator auto() const noexcept( conversion_is_noexcept() )
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          accessor_result_t< self_type > >
          ( *static_cast< const base_type* >( this ) );
      }
      else
      {
        return dr_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          extents_type,
                          typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_alloc< value_type >,
                          accessor_result_t< self_type > >
          ( *static_cast< const base_type* >( this ), allocator_result< const self_type >::get_allocator( ::std::forward< const self_type >( *this ) ) );
      }
    }
private:
  Tensor                          t_;
  [[no_unique_address]] Transpose indices_;
};

// Conjugate

template < class Tensor, class Transpose >
class conjugate_tensor_expression_traits
{
  public:
    using value_type   = decltype( ::std::conj( ::std::declval< typename ::std::remove_reference_t< Tensor >::value_type >() ) );
    using index_type   = typename ::std::remove_reference_t< Tensor >::index_type;
    using size_type    = typename ::std::remove_reference_t< Tensor >::size_type;
    using extents_type = typename transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::extents_type;
    using rank_type    = typename ::std::remove_reference_t< Tensor >::rank_type;
};

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
class conjugate_tensor_expression : public unary_tensor_expression_base< conjugate_tensor_expression< Tensor, Transpose >, conjugate_tensor_expression_traits< Tensor, Transpose > >
#else
template < class Tensor, class Transpose, typename Enable >
class conjugate_tensor_expression : public unary_tensor_expression_base< conjugate_tensor_expression< Tensor, Transpose, Enable >, conjugate_tensor_expression_traits< Tensor, Transpose > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = conjugate_tensor_expression< Tensor, Transpose >;
    #else
    using self_type   = conjugate_tensor_expression< Tensor, Transpose, Enable >;
    #endif
    using traits_type = conjugate_tensor_expression_traits< Tensor, Transpose >;
    using base_type   = unary_tensor_expression_base< self_type, traits_type >;
  public:
    // Special member functions
    constexpr conjugate_tensor_expression( Tensor&& t, Transpose&& indices = Transpose() ) noexcept : t_(t), indices_(indices) { }
    constexpr conjugate_tensor_expression& operator = ( const conjugate_tensor_expression& t ) noexcept { this->t_ = t.t_; this->indices_ = t.indices_; }
    constexpr conjugate_tensor_expression& operator = ( conjugate_tensor_expression&& t ) noexcept { this->t_ = t.t_;; this->indices_ = t.indices_; }
    // Aliases
    using value_type   = typename traits_type::value_type;
    using index_type   = typename traits_type::index_type;
    using size_type    = typename traits_type::size_type;
    using extents_type = typename traits_type::extents_type;
    using rank_type    = typename traits_type::rank_type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< Tensor >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::extents( this->t_.extents(), this->indices_ ); }
    [[nodiscard]] constexpr size_type extent( rank_type n ) const noexcept { return transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::extent( this->t_.extents(), this->indices_, n ); }
    // Unary tensor expression function
    [[nodiscard]] constexpr const Tensor& underlying() const noexcept { return this->t_; }
    // Conjugate
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator[]( OtherIndexType ... indices ) const noexcept( noexcept( ::std::conj( transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ) ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
    {
      if constexpr ( ::std::remove_reference_t< Tensor >::rank() == 1 )
      {
        return ::std::conj( LINALG_DETAIL::access( this->t_, indices ... ) );
      }
      else
      {
        return ::std::conj( transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ) );
      }
    }
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... OtherIndexType >
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( ::std::conj( transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ) ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...( OtherIndexType ) == rank() ) && ( ::std::is_convertible_v< OtherIndexType, index_type > && ... )
    #endif
    {
      if constexpr ( ::std::remove_reference_t< Tensor >::rank() == 1 )
      {
        return ::std::conj( LINALG_DETAIL::access( this->t_, indices ... ) );
      }
      else
      {
        return ::std::conj( transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >::access( this->t_, this->indices_, indices ... ) );
      }
    }
    #endif
  private:
    // Define noexcept specification of conversion operator
    [[nodiscard]] static inline constexpr bool conversion_is_noexcept() noexcept
    {
      if constexpr( extents_type::rank_dynamic() == 0 )
      {
        return ::std::is_nothrow_constructible_v< fs_tensor< value_type,
                                                             extents_type,
                                                             layout_result_t< self_type >,
                                                             accessor_result_t< self_type > >,
                                                  const base_type& >;
      }
      else
      {
        return ::std::is_nothrow_constructible_v< dr_tensor< value_type,
                                                             extents_type,
                                                             layout_result_t< self_type >,
                                                             extents_type,
                                                             typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_alloc< value_type >,
                                                             accessor_result_t< self_type > >,
                                                  const base_type&,
                                                  decltype( allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
      }
    }
  public:
    // Implicit conversion
    [[nodiscard]] constexpr operator auto() const noexcept( conversion_is_noexcept() )
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return fs_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          accessor_result_t< self_type > >
          ( *static_cast< const base_type* >( this ) );
      }
      else
      {
        return dr_tensor< value_type,
                          extents_type,
                          layout_result_t< self_type >,
                          extents_type,
                          typename ::std::allocator_traits< allocator_result_t< self_type > >::template rebind_alloc< value_type >,
                          accessor_result_t< self_type > >
          ( *static_cast< const base_type* >( this ), allocator_result< const self_type >::get_allocator( ::std::forward< const self_type >( *this ) ) );
      }
    }
private:
  Tensor                          t_;
  [[no_unique_address]] Transpose indices_;
};

LINALG_EXPRESSIONS_END // end expressions namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP
