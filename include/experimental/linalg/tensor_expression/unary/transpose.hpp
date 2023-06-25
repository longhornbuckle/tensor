//==================================================================================================
//  File:       transpose.hpp
//
//  Summary:    This header defines:
//              LINALG::accessor_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
//              LINALG::allocator_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
//              LINALG::layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
//              LINALG::is_alias_assignable< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
//              LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 >
//              LINALG_EXPRESSIONS_DETAIL::transpose_indices_v
//              LINALG_EXPRESSIONS_DETAIL::transpose_helper< Extents, TransposeIndices >
//              LINALG_EXPRESSIONS_DETAIL::transpose_tensor_expression_traits< Tensor, Transpose >
//              LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose >
//              LINALG::trans( const T& t )
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP
#define LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TENSOR_EXPRESSIONS_HPP

#include <experimental/linear_algebra.hpp>

LINALG_BEGIN // linalg namespace

//-------------------
//  Accessor Result
//-------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = rebind_accessor_t< typename ::std::remove_reference_t< Tensor >::accessor_type, ::std::remove_cv_t< typename ::std::remove_reference_t< Tensor >::value_type > >;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct accessor_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::accessor_type;
};

#else

template < class Tensor,
           class Transpose,
           class Enable >
struct accessor_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable > >
{
private:
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = typename ::std::conditional_t< LINALG_DETAIL::is_default_accessor_v< typename ::std::remove_reference_t< T >::accessor_type >,
                                                rebind_accessor< typename ::std::remove_reference_t< T >::accessor_type, ::std::remove_cv_t< typename ::std::remove_reference_t< T >::value_type > >,
                                                invalid_helper< T > >::type;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< T >().operator auto() )::accessor_type;
  };
public:
  using type = typename ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                              readable_helper< Tensor >,
                                              ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                    unevaluated_helper< Tensor >,
                                                                    invalid_helper< Tensor > > >::type;
};

#endif

//--------------------
//  Allocator Result
//--------------------

#ifdef LINALG_ENABLE_CONCEPTS

template < class Tensor,
           class Transpose >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
struct allocator_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = typename allocator_result< Tensor >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose >& t ) noexcept
    { return allocator_result< Tensor >::get_allocator( t.underlying() ); }
};

#else

template < class Tensor,
           class Transpose,
           class Enable >
struct allocator_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable > >
{
  using type = typename allocator_result< Tensor >::type;
  [[nodiscard]] static inline constexpr type get_allocator( const LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable >& t ) noexcept
  {
    return allocator_result< Tensor >::get_allocator( t.underlying() );
  }
};

#endif

//-----------------
//  Layout Result
//-----------------

#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose >
  requires ( LINALG_CONCEPTS::readable_tensor< ::std::remove_reference_t< Tensor > > &&
             ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
               ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ) )
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = LINALG::default_layout;
};

template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > >
{
  using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
};
#else
template < class Tensor, class Transpose, class Enable >
struct layout_result< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose, Enable > >
{
private :
  template < class T >
  struct invalid_helper;
  template < class T >
  struct readable_helper
  {
    using type = LINALG::default_layout;
  };
  template < class T >
  struct unevaluated_helper
  {
    using type = typename decltype( ::std::declval< Tensor >().operator auto() )::layout_type;
  };
public :
  using type = typename ::std::conditional_t< ( ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_right > ||
                                                ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_left > ||
                                                ::std::is_same_v< typename ::std::remove_reference_t< Tensor >::layout_type, ::std::layout_stride > ),
                                              ::std::conditional_t< LINALG_CONCEPTS::readable_tensor_v< ::std::remove_reference_t< Tensor > >,
                                                                    readable_helper< Tensor >,
                                                                   ::std::conditional_t< LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
                                                                                         unevaluated_helper< Tensor >,
                                                                                         invalid_helper< Tensor > > >,
                                     invalid_helper< Tensor > >::type;
};
#endif

//-----------------------
//  Is Alias Assignable
//-----------------------
template < class Tensor, class Transpose >
struct is_alias_assignable< LINALG_EXPRESSIONS::transpose_tensor_expression< Tensor, Transpose > > :
  public ::std::conditional_t< 
#ifdef LINALG_ENABLE_CONCEPTS
                               LINALG_CONCEPTS::unevaluated_tensor_expression< ::std::remove_reference_t< Tensor > >,
#else
                               LINALG_CONCEPTS::unevaluated_tensor_expression_v< ::std::remove_reference_t< Tensor > >,
#endif
                               is_alias_assignable< Tensor >,
                               ::std::true_type >
{ };

LINALG_END // linalg namespace

LINALG_EXPRESSIONS_DETAIL_BEGIN // expressions namespace

/// @brief Defines indices to be transposed via template type
template < ::std::size_t index1, ::std::size_t index2 >
struct transpose_indices_t
{
  [[nodiscard]] constexpr ::std::size_t first() const noexcept { return index1; }
  [[nodiscard]] constexpr ::std::size_t second() const noexcept { return index2; }
};

/// @brief Defines indices to be transposed via runtime values
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

/// @brief  Helper class for defining index transpose
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
        if ( n == static_cast< typename extents_type::rank_type >( transpose_indices.first() ) )
        {
          return e.extent( transpose_indices.second() );
        }
        else if ( n == static_cast< typename extents_type::rank_type >( transpose_indices.second() ) )
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

/// @brief Traits class for transpose tensor expression
/// @tparam Tensor underlying tensor to be transposed
/// @tparam Transpose conveys indices to be transposed
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

LINALG_EXPRESSIONS_DETAIL_END // expressions detail namespace

LINALG_EXPRESSIONS_BEGIN // expressions namespace

/// @brief Defines the transpose operation on a tensor
/// @tparam Tensor underlying tensor to be transposed
/// @tparam Transpose conveys indices to be transposed
#ifdef LINALG_ENABLE_CONCEPTS
template < class Tensor, class Transpose >
  requires LINALG_CONCEPTS::tensor_expression< ::std::remove_reference_t< Tensor > >
class transpose_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< transpose_tensor_expression< Tensor, Transpose >,
                                                                  LINALG_EXPRESSIONS_DETAIL::transpose_tensor_expression_traits< Tensor, Transpose > >
#else
template < class Tensor, class Transpose, typename Enable >
class transpose_tensor_expression :
  public LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< transpose_tensor_expression< Tensor, Transpose, Enable >,
                                                                  LINALG_EXPRESSIONS_DETAIL::transpose_tensor_expression_traits< Tensor, Transpose > >
#endif
{
  private:
    #ifdef LINALG_ENABLE_CONCEPTS
    using self_type   = transpose_tensor_expression< Tensor, Transpose >;
    #else
    using self_type   = transpose_tensor_expression< Tensor, Transpose, Enable >;
    #endif
    using traits_type = LINALG_EXPRESSIONS_DETAIL::transpose_tensor_expression_traits< Tensor, Transpose >;
    using base_type   = LINALG_EXPRESSIONS_DETAIL::unary_tensor_expression_base< self_type, traits_type >;
    using helper_type = LINALG_EXPRESSIONS_DETAIL::transpose_helper< typename ::std::remove_reference_t< Tensor >::extents_type, Transpose >;
  public:
    // Special member functions
    constexpr transpose_tensor_expression( Tensor&& t, Transpose&& indices = Transpose() ) noexcept : t_(t), indices_(indices) { }
    constexpr transpose_tensor_expression& operator = ( const transpose_tensor_expression& t ) noexcept { this->t_ = t.t_; this->indices_ = t.indices_; }
    constexpr transpose_tensor_expression& operator = ( transpose_tensor_expression&& t ) noexcept { this->t_ = t.t_;; this->indices_ = t.indices_; }
    // Aliases
    using value_type     = typename traits_type::value_type;
    using index_type     = typename traits_type::index_type;
    using size_type      = typename traits_type::size_type;
    using extents_type   = typename traits_type::extents_type;
    using rank_type      = typename traits_type::rank_type;
  private:
    template < class T, bool >
    struct helper
    {
      using type = LINALG::fs_tensor< typename T::value_type,
                                      typename T::extents_type,
                                      LINALG::layout_result_t< T >,
                                      LINALG::accessor_result_t< T > >;
    };
    template < class T >
    struct helper< T, false >
    {
      using type = LINALG::dr_tensor< typename T::value_type,
                                      typename T::extents_type,
                                      LINALG::layout_result_t< T >,
                                      typename T::extents_type,
                                      typename ::std::allocator_traits< LINALG::allocator_result_t< T > >::template rebind_alloc< typename T::value_type >,
                                      LINALG::accessor_result_t< T > >;
    };
  public:
    using evaluated_type = typename helper< self_type, ( extents_type::rank_dynamic() == 0 ) >::type;
    // Tensor expression functions
    [[nodiscard]] static constexpr rank_type rank() noexcept { return ::std::remove_reference_t< Tensor >::rank(); }
    [[nodiscard]] constexpr extents_type extents() const noexcept { return helper_type::extents( this->t_.extents(), this->indices_ ); }
    [[nodiscard]] constexpr index_type extent( rank_type n ) const noexcept { return helper_type::extent( this->t_.extents(), this->indices_, n ); }
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
    [[nodiscard]] constexpr value_type operator()( OtherIndexType ... indices ) const noexcept( noexcept( helper_type::access( this->t_, this->indices_, indices ... ) ) )
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
        return helper_type::access( this->t_, this->indices_, indices ... );
      }
    }
    #endif
  private:
    // Define noexcept specification of conversion operator
    [[nodiscard]] static inline constexpr bool conversion_is_noexcept() noexcept
    {
      if constexpr( extents_type::rank_dynamic() == 0 )
      {
        return ::std::is_nothrow_constructible_v< evaluated_type,
                                                  const base_type& >;
      }
      else
      {
        return ::std::is_nothrow_constructible_v< evaluated_type,
                                                  const base_type&,
                                                  decltype( LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( ::std::declval< const self_type >() ) ) ) >;
      }
    }
  public:
    // Implicit conversion
    [[nodiscard]] constexpr operator evaluated_type() const noexcept( conversion_is_noexcept() )
    {
      if constexpr ( extents_type::rank_dynamic() == 0 )
      {
        return evaluated_type( *static_cast< const base_type* >( this ) );
      }
      else
      {
        return evaluated_type( *static_cast< const base_type* >( this ), LINALG::allocator_result< self_type >::get_allocator( ::std::forward< const self_type >( *this ) ) );
      }
    }
    // Evaluated expression
    [[nodiscard]] constexpr LINALG_FORCE_INLINE_FUNCTION auto evaluate() const noexcept( conversion_is_noexcept() )
    {
      return evaluated_type( *this );
    }
private:
  Tensor                          t_;
  [[no_unique_address]] Transpose indices_;
};

LINALG_EXPRESSIONS_END // expressions namespace

LINALG_BEGIN // linalg namespace


//----------------------------
//  Unary transpose operator
//----------------------------
#ifdef LINALG_ENABLE_CONCEPTS
template < class T >
#else
template < class T,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<> >,
                                                                                                                      const T&,
                                                                                                                      LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<> > &&
                                          ( LINALG_CONCEPTS::vector_expression_v< ::std::remove_reference_t< T > > ||
                                            LINALG_CONCEPTS::matrix_expression_v< ::std::remove_reference_t< T > > ) > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const T& t ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<> >,
                                                                                         const T&,
                                                                                         LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<> > &&
             ( ::std::remove_reference_t< T >::rank() < 3 ) )
#endif
{
  return LINALG_EXPRESSIONS::transpose_tensor_expression< const T& >( t, LINALG_EXPRESSIONS_DETAIL::transpose_indices_t<> {} );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, auto index1, auto index2 >
#else
template < class T, auto index1, auto index2,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 > >,
                                                                                                                      const T&,
                                                                                                                      LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 > > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const T& t, [[maybe_unused]] const LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 >& indices = LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 > {} ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 > >,
                                                                                       const T&,
                                                                                       LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 > >
#endif
{
  return LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS_DETAIL::transpose_indices_t< index1, index2 > >( t, indices );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, class IndexType >
#else
template < class T, class IndexType,
           typename = ::std::enable_if_t< ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS_DETAIL::transpose_indices_v< IndexType, IndexType > >,
                                                                     const T&,
                                                                     LINALG_EXPRESSIONS_DETAIL::transpose_indices_v< IndexType, IndexType > > > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const T& t, IndexType index1, IndexType index2 ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< LINALG_EXPRESSIONS::transpose_tensor_expression< const T&, LINALG_EXPRESSIONS_DETAIL::transpose_indices_v< IndexType, IndexType > >,
                                                                                       const T&,
                                                                                       LINALG_EXPRESSIONS_DETAIL::transpose_indices_v< IndexType, IndexType > >
#endif
{
  return LINALG_EXPRESSIONS::transpose_tensor_expression< const T&,LINALG_EXPRESSIONS_DETAIL::transpose_indices_v< IndexType, IndexType > >
    ( t, LINALG_EXPRESSIONS_DETAIL::transpose_indices_v< IndexType, IndexType > { index1, index2 } );
}

LINALG_END // linalg namesapce

#endif  //- LINEAR_ALGEBRA_TENSOR_EXPRESSION_UNARY_TRANSPOSE_HPP
