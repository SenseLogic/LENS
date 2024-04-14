## -- IMPORTS

import cv2
import numpy
import onnxruntime
import sys

## -- FUNCTIONS

def GetLogicalPath( path ) :

    return path.replace( '\\', '/' );

## ~~

def GetPreprocessedImage( image ) :

    image = numpy.transpose( image[ :, :, 0:3 ], ( 2, 0, 1 ) );
    image = numpy.expand_dims( image, axis = 0 ).astype( numpy.float32 );

    return image;

## ~~

def GetPostProcessedImage( image ) :

    image = numpy.squeeze( image );
    image = numpy.transpose( image, ( 1, 2, 0 ) ).astype( numpy.uint8 );

    return image;

## ~~

def GetUpscaledImageData( upscaling_model_path, image_array ) :

    session_options = onnxruntime.SessionOptions();
    session_options.intra_op_num_threads = 1;
    session_options.inter_op_num_threads = 1;
    session = onnxruntime.InferenceSession( upscaling_model_path, session_options );

    input_array = { session.get_inputs()[ 0 ].name: image_array };
    output_array = session.run( None, input_array );

    return output_array[ 0 ];

## ~~

def UpscaleImage( input_image_path, output_image_path, upscaling_model_path ) :

    print( "Loading image :", input_image_path );
    image = cv2.imread( input_image_path );

    if image is None :

        raise ValueError( "Image could not be read." );

    if image.ndim == 2 :

        image = cv2.cvtColor( image, cv2.COLOR_GRAY2BGR );

    if image.shape[ 2 ] == 4 :

        alpha = image[ :, :, 3 ];
        alpha = cv2.cvtColor( alpha, cv2.COLOR_GRAY2BGR );
        alpha_output = GetPostProcessedImage( GetUpscaledImageData( upscaling_model_path, GetPreprocessedImage( alpha ) ) );
        alpha_output = cv2.cvtColor( alpha_output, cv2.COLOR_BGR2GRAY );

        image = image[ :, :, 0:3 ];
        image_output = GetPostProcessedImage( GetUpscaledImageData( upscaling_model_path, GetPreprocessedImage( image ) ) );
        image_output = cv2.cvtColor( image_output, cv2.COLOR_BGR2BGRA );
        image_output[ :, :, 3 ] = alpha_output;

    elif image.shape[ 2 ] == 3 :

        image_output = GetPostProcessedImage( GetUpscaledImageData( upscaling_model_path, GetPreprocessedImage( image ) ) );

    print( "Saving image :", output_image_path );
    cv2.imwrite( output_image_path, image_output );

## -- STATEMENTS

argument_array = sys.argv;
argument_count = len( argument_array ) - 1;

if ( argument_count == 3 ) :

    upscaling_model_path = GetLogicalPath( argument_array[ 1 ] );
    input_image_path = GetLogicalPath( argument_array[ 2 ] );
    output_image_path = GetLogicalPath( argument_array[ 3 ] );

    UpscaleImage( input_image_path, output_image_path, upscaling_model_path );

    sys.exit( 0 );

print( f"*** Invalid arguments : { argument_array }" );
print( "Usage: python nano.py image.png upscaled_image.png" );
