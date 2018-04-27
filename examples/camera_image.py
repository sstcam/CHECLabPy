import numpy as np
from matplotlib import pyplot as plt
from CHECLabPy.core.io import DL1Reader, TIOReader
from CHECLabPy.plotting.camera import CameraImage, CameraImageImshow
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping, \
    get_superpixel_mapping, get_tm_mapping


def plot_from_tio():
    """
    Use the CHECLabPy mapping dataframe to plot an image
    """
    path = "/Users/Jason/Software/CHECLabPy/refdata/Run17473_r1.tio"
    r = TIOReader(path, max_events=10)
    camera = CameraImage.from_mapping(r.mapping)
    camera.add_colorbar("Amplitude (mV)")
    camera.annotate_on_telescope_up()
    for wf in r:
        image = wf[:, 60]
        camera.image = image
        plt.pause(0.5)


def plot_from_dl1():
    """
    Use the CHECLabPy mapping dataframe to plot an image
    """
    path = "/Users/Jason/Software/CHECLabPy/refdata/Run17473_dl1.h5"
    r = DL1Reader(path)
    camera = CameraImage.from_mapping(r.mapping)
    camera.add_colorbar("Charge (mV ns)")
    for i, df in enumerate(r.iterate_over_events()):
        if i > 10:
            break
        charge = df['charge'].values
        camera.image = charge
        plt.pause(0.1)


def plot_from_coordinates():
    """
    Plot directly with coordinates
    """
    from target_calib import CameraConfiguration
    c = CameraConfiguration("1.0.1")
    m = c.GetMapping()
    xpix = np.array(m.GetXPixVector())
    ypix = np.array(m.GetYPixVector())
    size = m.GetSize()
    camera = CameraImage(xpix, ypix, size)
    image = np.zeros(xpix.size)
    image[::2] = 1
    camera.image = image
    plt.show()


def plot_from_tc_mapping():
    """
    Plot using the TargetCalib Mapping class
    """
    from target_calib import CameraConfiguration
    c = CameraConfiguration("1.0.1")
    m = c.GetMapping()
    camera = CameraImage.from_tc_mapping(m)
    image = np.zeros(m.GetNPixels())
    image[::2] = 1
    camera.image = image
    plt.show()


def plot_from_camera_version():
    """
    Plot by specifying a camera version (requires TargetCalib)
    """
    camera_version = "1.0.1"
    camera = CameraImage.from_camera_version(camera_version)
    image = np.zeros(2048)
    image[::2] = 1
    camera.image = image
    plt.show()


def plot_from_camera_version_single_module():
    """
    Plot a single module by specifying a camera version (requires TargetCalib)
    """
    camera_version = "1.0.1"
    camera = CameraImage.from_camera_version(camera_version, True)
    image = np.zeros(64)
    image[::2] = 1
    camera.image = image
    plt.show()


def plot_imshow():
    """
    Plot the camera image using imshow (essentially a 2D histogram). Therefore
    does not include module gaps
    """
    r = DL1Reader("/Users/Jason/Software/CHECLabPy/refdata/Run17473_dl1.h5")
    camera = CameraImageImshow.from_mapping(r.mapping)
    camera.add_colorbar("Charge (mV ns)")
    for df in r.iterate_over_events():
        charge = df['charge'].values
        camera.image = charge
        plt.pause(0.1)


def plot_superpixel():
    """
    Make a camera image plot for values that are per superpixel
    """
    from target_calib import CameraConfiguration
    c = CameraConfiguration("1.0.1")
    m = c.GetMapping()
    df = get_superpixel_mapping(get_clp_mapping_from_tc_mapping(m))
    camera = CameraImage.from_mapping(df)
    image = np.zeros(m.GetNSuperPixels())
    image[::2] = 1
    camera.image = image
    plt.show()


def plot_tm():
    """
    Make a camera image plot for values that are per superpixel
    """
    from target_calib import CameraConfiguration
    c = CameraConfiguration("1.0.1")
    m = c.GetMapping()
    df = get_tm_mapping(get_clp_mapping_from_tc_mapping(m))
    camera = CameraImage.from_mapping(df)
    image = np.zeros(m.GetNModules())
    image[::2] = 1
    camera.image = image
    plt.show()


def plot_pixel_postions():
    """
    Plot pixel positions onto the camera
    """
    camera_version = "1.0.1"
    camera = CameraImage.from_camera_version(camera_version)
    pixels = np.arange(camera.n_pixels)
    camera.add_pixel_text(pixels)
    plt.show()


def plot_tm_edge_labels():
    """
    Annotate plot with the TM numbers on the edges
    """
    camera_version = "1.0.1"
    camera = CameraImage.from_camera_version(camera_version)
    pixels = np.arange(camera.n_pixels)
    camera.annotate_tm_edge_label()
    plt.show()


def plot_tm_edge_labels_imshow():
    """
    Annotate plot with the TM numbers on the edges
    """
    camera_version = "1.0.1"
    camera = CameraImageImshow.from_camera_version(camera_version)
    pixels = np.arange(camera.n_pixels)
    camera.annotate_tm_edge_label()
    plt.show()


if __name__ == '__main__':
    plot_from_tio()
    plot_from_dl1()
    plot_from_coordinates()
    plot_from_tc_mapping()
    plot_from_camera_version()
    plot_from_camera_version_single_module()
    plot_imshow()
    plot_superpixel()
    plot_tm()
    plot_pixel_postions()
    plot_tm_edge_labels()
    plot_tm_edge_labels_imshow()
