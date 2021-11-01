###
# title: pysci.imagine.py
#
# language Python3
# license: GPLv3
# author: bue
# date: 2019-01-31
#
# run:
#    form pysci import imagine
#
# description:
#    my image analysis library
####

# library
import numpy as np
import pandas as pd

# function
def slide_up(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one row up.
        top row get deleted,
        bottom row of zeros is inserted.

    description:
      inspired by np.roll function, though elements that roll
      beyond the last position are not re-introduced at the first.
    """
    a = np.delete(np.insert(a, -1, 0, axis=0), 0, axis=0)
    return(a)


def slide_down(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one row down.
        top row of zeros is inserted.
        bottom row get deleted,

    description:
      inspired by np.roll function, though elements that roll
      beyond the last position are not re-introduced at the first.
    """
    a = np.delete(np.insert(a, 0, 0, axis=0), -1, axis=0)
    return(a)


def slide_left(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one column left.
        left most column gets deleted,
        right most a column of zeros is inserted.

    description:
      inspired by np.roll function, though elements that roll
      beyond the last position are not re-introduced at the first.
    """
    a = np.delete(np.insert(a, -1, 0, axis=1), 0, axis=1)
    return(a)


def slide_right(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one column right.
        left most a column of zeros is inserted.
        right most column gets deleted,

    description:
      inspired by np.roll function, though elements that roll
      beyond the last position are not re-introduced at the first.
    """
    a = np.delete(np.insert(a, 0, 0, axis=1), -1, axis=1)
    return(a)


def slide_upleft(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one row up and one column left.

    description:
      inspired by np.roll function.
    """
    a = slide_left(slide_up(a))
    return(a)


def slide_upright(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one row up and one column right.

    description:
      inspired by np.roll function.
    """
    a = slide_right(slide_up(a))
    return(a)


def slide_downleft(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one row down  and one column left.

    description:
      inspired by np.roll function.
    """
    a = slide_left(slide_down(a))
    return(a)


def slide_downright(a):
    """
    input:
      a: numpy array

    output:
      a: input numpy array shifted one row down and one column right.

    description:
      inspired by np.roll function.
    """
    a = slide_right(slide_down(a))
    return(a)



def get_border(ai_basin):
    """
    input:
      ai_basin: numpy array representing a cells or nuclei basin file.
        it is assumed that basin borders are represented by 0 values,
        and basins are represented with any values different from 0.
        ai_basin = skimage.io.imread("cells_basins.tif")

    output:
      ai_border: numpy array containing only the cell or nuclei basin border.
        border value will be 1, non border value will be 0.

    description:
      algorithm to extract the basin borders form basin numpy arrays.
    """
    ab_border_up = (ai_basin - slide_up(ai_basin)) != 0
    ab_border_down = (ai_basin - slide_down(ai_basin)) != 0
    ab_border_left = (ai_basin - slide_left(ai_basin)) != 0
    ab_border_right = (ai_basin - slide_right(ai_basin)) != 0
    ab_border_upleft = (ai_basin - slide_upleft(ai_basin)) != 0
    ab_border_upright = (ai_basin - slide_upright(ai_basin)) != 0
    ab_border_downleft = (ai_basin - slide_downleft(ai_basin)) != 0
    ab_border_downright = (ai_basin - slide_downright(ai_basin)) != 0
    ab_border = ab_border_up | ab_border_down | ab_border_left | ab_border_right | ab_border_upleft | ab_border_upright | ab_border_downleft | ab_border_downright 
    ai_border = ab_border * 1
    return(ai_border)


def collision(ai_basin, i_step_size=1):
    """
    input:
      ai_basin: numpy array representing a cells basin file.
        it is assumed that basin borders are represented by 0 values,
        and basins are represented with any values different from 0.
        ai_basin = skimage.io.imread("cells_basins.tif")

    i_step_size: integer that specifies the distance from a basin
        where collisions with other basins are detected.
        increasing the step size behind > 1 will result in faster processing
        but less certain results. step size < 1 make no sense.
        default step size is 1.

    output:
        eti_collision: a set of tuples representing colliding basins.

    description:
        algorithm to detect which basin collide a given step size away.
    """
    eti_collision = set()
    for o_slide in {slide_up, slide_down, slide_left, slide_right, slide_upleft, slide_upright, slide_downleft, slide_downright}:
        ai_walk = ai_basin.copy()
        for _ in range(i_step_size):
            ai_walk = o_slide(ai_walk)
        ai_alice = ai_walk[(ai_basin != 0) & (ai_walk != 0)]
        ai_bob = ai_basin[(ai_basin != 0) & (ai_walk != 0)]
        eti_collision = eti_collision.union(set(
            zip(
                ai_alice[(ai_alice != ai_bob)],
                ai_bob[(ai_bob != ai_alice)]
            )
        ))
    # return
    return(eti_collision)


def grow(ai_basin, i_step=1):
    """
    input:
      ai_basin: numpy array representing a cells basin file.
        it is assumed that basin borders are represented by 0 values,
        and basins are represented with any values different from 0.
        ai_basin = skimage.io.imread("cells_basins.tif")

      i_step: integer which specifies how many pixels the basin
        to  each direction should grow

    output:
      ai_grown: numpy array with the grown basins

    description:
      algorithm to grow the basis in a given basin numpy array.
      growing happens counterclockwise.
    """
    ai_grown = ai_basin.copy()
    for _ in range(i_step):
        for o_slide in {slide_up, slide_upleft, slide_left, slide_downleft, slide_down, slide_downright, slide_right, slide_upright}:
            ai_alice = ai_basin.copy()
            ai_evolve = o_slide(ai_alice)
            ai_alice[(ai_evolve != ai_alice) & (ai_alice == 0)] = ai_evolve[(ai_evolve != ai_alice) & (ai_alice == 0)]
            # update grown
            ai_grown[(ai_alice != ai_grown) & (ai_grown == 0)] = ai_alice[(ai_alice != ai_grown) & (ai_grown == 0)]
    # output
    return(ai_grown)


def touching_cells(ai_basin, i_border_width=0, i_step_size=1):
    """
    input:
      ai_basin: numpy array representing a cells basin file.
        it is assumed that basin borders are represented by 0 values,
        and basins are represented with any values different from 0.
        ai_basin = skimage.io.imread("cells_basins.tif")

      i_border_width: maximal acceptable border with in pixels.
        this is half of the range how far two the adjacent cell maximal
        can be apart and still are regarded as touching each other.

      i_step_size: step size by which the border width is sampled for
        touching cells.
        increase the step size behind > 1 will result in faster processing
        but less certain results. step size < 1 make no sense.
        default step size is 1.

    output:
      dei_touch: a dictionary that for each basin states
        which other basins are touching.

    description:
      algorithm to extract the touching basins from a cell basin numpy array.
      algorithm inspired by C=64 computer games with sprit collision.
    """

    # detect neighbors
    eti_collision = set()
    ai_evolve = ai_basin.copy()
    for _ in range(-1, i_border_width, i_step_size):
        # detect cell border collision
        eti_collision = eti_collision.union(
            collision(ai_basin=ai_evolve, i_step_size=i_step_size)
        )
        # grow basin
        ai_evolve = grow(ai_basin=ai_evolve, i_step=i_step_size)

    # transform set of tuple of alice and bob collision to dictionary of sets
    dei_touch = {}
    ei_alice = set(np.ndarray.flatten(ai_basin))
    ei_alice.remove(0)
    for i_alice in ei_alice:
        dei_touch.update({i_alice : set()})
    for i_alice, i_bob in eti_collision:
        ei_bob = dei_touch[i_alice]
        ei_bob.add(i_bob)
        dei_touch.update({i_alice : ei_bob})

    # output
    return(dei_touch)


def detouch2df(deo_abc, ls_column=["cell_center","cell_touch"]):
    """
    input:
        deo_touch: touching_cells generated dictionary
        ls_column: future dictionary_key dictionary_value column name

    output:
        df_touch: dataframe which contains the same information
          as the input deo_touch dictionary.

    description:
        transforms dei_touch dictionary into a two column dataframe.
    """
    lo_key_total= []
    lo_value_total = []
    for o_key, eo_value in deo_abc.items():
        try:
            lo_value = sorted(eo_value, key=int)
        except ValueError:
            lo_value = sorted(eo_value)
        # extract form dictionary
        if (len(lo_value) == 0):
            lo_key_total.append(o_key)
            lo_value_total.append(0)
        else:
            lo_key_total.extend([o_key] * len(lo_value))
            lo_value_total.extend(lo_value)
    # generate datafarme
    df_touch = pd.DataFrame([lo_key_total,lo_value_total], index=ls_column).T
    return(df_touch)


def imgfuse(laaai_in):
    """
    input:
        laaai_in: list of 3 channel (RGB) images

    output:
       aaai_out: fused 3 channel image

    description:
       code to fuse many RGB images into one.
    """
    # check shape
    ti_shape = None
    for aaai_in in laaai_in:
        if (ti_shape is None):
            ti_shape = aaai_in.shape
        else:
           if (aaai_in.shape != ti_shape):
               sys.exit(f"Error: input images have not the same shape. {aaai_in.shape} != {aaai_in}.")

    # fuse images
    llli_channel = []
    for i_channel in range(ti_shape[0]):
        lli_matrix = []
        for i_y in range(ti_shape[1]):
            li_row = []
            for i_x in range(ti_shape[2]):
                #print(f"{i_channel} {i_y} {i_x}")
                li_px = []
                for aaai_in in laaai_in:
                    i_in = aaai_in[i_channel,i_y,i_x]
                    if (i_in != 0):
                        li_px.append(i_in)
                if (len(li_px) != 0):
                    i_out = np.mean(li_px)
                else:
                    i_out = 0
                li_row.append(int(i_out))
            lli_matrix.append(li_row)
        llli_channel.append(lli_matrix)

    # output
    aaai_out = np.array(llli_channel)
    return(aaai_out)



# test code
if __name__ == "__main__":

    # load basins tiff into numpy array
    '''
    import matplotlib.pyplot as plt
    import skimage as ski
    a_tiff = ski.io.imread("cells_basins.tif")
    plt.imshow(a_tiff)
    '''

    # generate test data
    a = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,4,0,0,0],
        [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,2,2,2,0,0,0],
        [0,0,0,0,3,3,3,0,2,2,2,0,0,0],
        [0,0,0,0,3,3,3,0,2,2,2,0,0,0],
        [0,0,0,0,3,3,3,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    b = np.array([
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,1,2,0,0,0,0,0],
        [0,0,0,0,0,1,2,0,0,0,0],
        [0,0,0,0,0,0,0,2,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ])

    c = np.array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ])

    # run get_border
    print("\nborderwall_tm")
    print(a)
    print(get_border(a))
    #plt.imshow(get_border(a_tiff))

    # run grow
    '''
    print("\ngrow")
    print(c)
    print(grow(c))
    print(grow(grow(c)))
    print(grow(c, i_step_size=2))
    print(b)
    print(grow(b))
    print(grow(grow(b)))
    print(grow(b, i_step_size=2))
    '''

    # run collision
    '''
    print("\ncollision")
    print(c)
    print(collision(c))
    print(b)
    print(collision(b))
    print(c)
    print(collision(c))
    '''

    # run touching_cells
    print("\ntouch")
    #print(a)
    print(touching_cells(a, i_border_width=0))
    print(touching_cells(a, i_border_width=1))
    print(touching_cells(a, i_border_width=2))
    print(touching_cells(a, i_border_width=3))
    print(touching_cells(a, i_border_width=4))
    print(touching_cells(a, i_border_width=4, i_step_size=2))
    #touching_cells(a_tiff, i_border_width=1)


    # img fuse
    aaai_1 = np.array([
        [[1,1,1],[2,2,2],[3,3,3]],
        [[0,0,0,],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,0]],
    ])
    aaai_2 = np.array([
        [[0,0,0,],[0,0,0],[0,0,0]],
        [[1,1,1],[2,2,2],[3,3,3]],
        [[0,0,0],[0,0,0],[0,0,0]],
    ])
    aaai_3 = np.array([
        [[0,0,0,],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,0]],
        [[1,1,1],[2,2,2],[3,3,3]],
    ])
    aaai_4 = np.array([
        [[1,1,1],[2,2,2],[3,3,3]],
        [[1,1,1],[2,2,2],[3,3,3]],
        [[0,0,0],[0,0,0],[0,0,0]],
    ])
    aaai_5 = np.array([
        [[0,0,0,],[0,0,0],[0,0,0]],
        [[1,1,1],[2,2,2],[3,3,3]],
        [[1,1,1],[2,2,2],[3,3,3]],
    ])
    aaai_out = imgfuse([aaai_1, aaai_2, aaai_3, aaai_4, aaai_5])
    print("fused 3channel image:\n", aaai_out, type(aaai_out))
