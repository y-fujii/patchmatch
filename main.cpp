#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cstdint>
#include <random>
#include <Eigen/Dense>
#include <stb_image.h>
#include <stb_image_write.h>

using namespace std;


template<class T, int N>
using Pixel = Eigen::Matrix<T, N, 1>;

template<class T>
using Image = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<class T, int Radius>
struct PatchMatcher {
	PatchMatcher( Image<T> const& img0, Image<T> const& img1 ):
		image0( img0 ),
		image1( img1 ),
		nnf( img0.rows(), img0.cols() ),
		score( img0.rows(), img0.cols() )
	{
		uniform_int_distribution<int> dist0( Radius, image1.rows() - Radius - 1 );
		uniform_int_distribution<int> dist1( Radius, image1.cols() - Radius - 1 );
		for( int y = Radius; y < image0.cols() - Radius; ++y ) {
			for( int x = Radius; x < image0.rows() - Radius; ++x ) {
				nnf( x, y )( 0 ) = dist0( rng );
				nnf( x, y )( 1 ) = dist1( rng );
				score( x, y ) = distance( x, y, nnf( x, y )( 0 ), nnf( x, y )( 1 ) );
			}
		}
	}

	int distance( int x0, int y0, int x1, int y1 ) {
		int s = 0;
		for( int dy = -Radius; dy <= +Radius; ++dy ) {
			for( int dx = -Radius; dx <= +Radius; ++dx ) {
				auto v0 = image0( x0 + dx, y0 + dy ).template cast<int>();
				auto v1 = image1( x1 + dx, y1 + dy ).template cast<int>();
				s += (v0 - v1).squaredNorm();
			}
		}
		return s;
	}

	void update( int x, int y, int x0, int y0 ) {
		if( Radius <= x0 && x0 < image1.rows() - Radius &&
		    Radius <= y0 && y0 < image1.cols() - Radius )
		{
			int d = distance( x, y, x0, y0 );
			if( d < score( x, y ) ) {
				nnf( x, y ) = { x0, y0 };
				score( x, y ) = d;
			}
		}
	}

	template<int D>
	void propagate( int x, int y ) {
		int x0 = nnf( x + D, y )( 0 ) - D;
		int y0 = nnf( x + D, y )( 1 );
		update( x, y, x0, y0 );

		int x1 = nnf( x, y + D )( 0 );
		int y1 = nnf( x, y + D )( 1 ) - D;
		update( x, y, x1, y1 );
	}

	void search( int x, int y ) {
		int x0 = nnf( x, y )( 0 );
		int y0 = nnf( x, y )( 1 );
		int r = max( image1.rows(), image1.cols() );
		while( r >= 1 ) {
			uniform_int_distribution<int> dist( -r, +r );
			int x1 = dist( rng ) + x0;
			int y1 = dist( rng ) + y0;
			update( x, y, x1, y1 );
			r >>= 1;
		}
	}

	void iterate() {
		for( int y = Radius; y < image0.cols() - Radius; ++y ) {
			for( int x = Radius; x < image0.rows() - Radius; ++x ) {
				propagate<-1>( x, y );
				search( x, y );
			}
		}
		for( int y = image0.cols() - Radius - 1; y >= Radius; --y ) {
			for( int x = image0.rows() - Radius - 1; x >= Radius; --x ) {
				propagate<+1>( x, y );
				search( x, y );
			}
		}
	}

	Image<T> const&        image0;
	Image<T> const&        image1;
	Image<Eigen::Vector2i> nnf;
	Image<int>             score;
	mt19937_64             rng;
};


template<int N>
Image<Pixel<uint8_t, N>> loadImage( string const& fn ) {
	int x, y, n;
	uint8_t* buf = stbi_load( fn.c_str(), &x, &y, &n, N );
	if( buf == nullptr ) {
		throw std::runtime_error( "" );
	}

	Image<Pixel<uint8_t, N>> img = Eigen::Map<Image<Pixel<uint8_t, N>>>(
		reinterpret_cast<Pixel<uint8_t, N>*>( buf ), x, y
	);

	stbi_image_free( buf );
	return img;
}

template<int N>
void saveImage( Image<Pixel<uint8_t, N>> const& img, string const& fn ) {
	if( stbi_write_png( fn.c_str(), img.rows(), img.cols(), N, img.data(), img.rows() * N ) == 0 ) {
		throw std::runtime_error( "" );
	}
}

int main() {
	auto src0 = loadImage<3>( "src0.png" );
	auto src1 = loadImage<3>( "src1.png" );

	PatchMatcher<Pixel<uint8_t, 3>, 3> pm( src0, src1 );
	for( int i = 0; i < 3; ++i ) {
		pm.iterate();
	}

	int w = pm.nnf.rows();
	int h = pm.nnf.cols();
	Image<Pixel<uint8_t, 3>> dst( w, h );
	for( int y = 0; y < h; ++y ) {
		for( int x = 0; x < w; ++x ) {
			dst( x, y ) = src1( pm.nnf( x, y )( 0 ), pm.nnf( x, y )( 1 ) );
			/*
			dst( x, y )( 0 ) = 255 * pm.nnf( x, y )( 0 ) / src1.rows();
			dst( x, y )( 1 ) = 255 * pm.nnf( x, y )( 1 ) / src1.cols();
			dst( x, y )( 2 ) = 128;
			*/
		}
	}

	saveImage( dst, "dst.png" );
	return 0;
}
