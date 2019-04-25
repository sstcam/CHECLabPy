import pandas as pd


class IlluminationProfile:
    def __init__(self, path):
        """
        Class to handle the unfolding of a illumination profile from an image

        Parameters
        ----------
        path : str
            Path to an illumination profile file
        """
        profile_df = pd.read_csv(path, sep='\t')
        self.profile = profile_df['correction'].values

    def unfold(self, image):
        """
        Remove the undlying illumination profile from an image

        Parameters
        ----------
        image : ndarray
            Image array of n_pixels

        Returns
        -------
        unfolded : ndarray
            Unfolded image
        """
        return image / self.profile

    def fold(self, image):
        """
        Add the illumination profile to an image

        Parameters
        ----------
        image : ndarray
            Image array of n_pixels

        Returns
        -------
        folded : ndarray
            Folded image
        """
        return image * self.profile
