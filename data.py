import numpy as np
import h5py


class Data:
    def __init__(self, folder, filename):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'

    def create_file(self,
                    distribution,
                    density, current_v, current_w,
                    electric_x, electric_y, electric_z,
                    magnetic_y, magnetic_z):
        # Open file for writing
        with h5py.File(self.write_filename, 'w') as f:
            # Create datasets, dataset_distribution =
            f.create_dataset('pdf', data=np.array([distribution]),
                             chunks=True,
                             maxshape=(None, distribution.shape[0],
                                       distribution.shape[1], distribution.shape[2],
                                       distribution.shape[3], distribution.shape[4],
                                       distribution.shape[5], distribution.shape[6]),
                             dtype='f')
            f.create_dataset('density', data=np.array([density]),
                             chunks=True,
                             maxshape=(None, density.shape[0]),
                             dtype='f')
            f.create_dataset('current_v', data=np.array([current_v]),
                             chunks=True,
                             maxshape=(None, current_v.shape[0]),
                             dtype='f')
            f.create_dataset('current_w', data=np.array([current_w]),
                             chunks=True,
                             maxshape=(None, current_w.shape[0]),
                             dtype='f')
            f.create_dataset('electric_x', data=np.array([electric_x]),
                             chunks=True,
                             maxshape=(None, electric_x.shape[0]),
                             dtype='f')
            f.create_dataset('electric_y', data=np.array([electric_y]),
                             chunks=True,
                             maxshape=(None, electric_y.shape[0]),
                             dtype='f')
            f.create_dataset('electric_z', data=np.array([electric_z]),
                             chunks=True,
                             maxshape=(None, electric_z.shape[0]),
                             dtype='f')
            f.create_dataset('magnetic_y', data=np.array([magnetic_y]),
                             chunks=True,
                             maxshape=(None, magnetic_y.shape[0]),
                             dtype='f')
            f.create_dataset('magnetic_z', data=np.array([magnetic_z]),
                             chunks=True,
                             maxshape=(None, magnetic_z.shape[0]),
                             dtype='f')
            f.create_dataset('time', data=[0.0], chunks=True, maxshape=(None,))
            f.create_dataset('total_energy', data=[], chunks=True, maxshape=(None,))
            f.create_dataset('total_density', data=[], chunks=True, maxshape=(None,))

    def save_data(self, distribution,
                  density, current_v, current_w,
                  electric_x, electric_y, electric_z,
                  magnetic_y, magnetic_z, time):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['pdf'].resize((f['pdf'].shape[0] + 1), axis=0)
            f['density'].resize((f['density'].shape[0] + 1), axis=0)
            f['current_v'].resize((f['current_v'].shape[0] + 1), axis=0)
            f['current_w'].resize((f['current_w'].shape[0] + 1), axis=0)
            f['electric_x'].resize((f['electric_x'].shape[0] + 1), axis=0)
            f['electric_y'].resize((f['electric_y'].shape[0] + 1), axis=0)
            f['electric_z'].resize((f['electric_z'].shape[0] + 1), axis=0)
            f['magnetic_y'].resize((f['magnetic_y'].shape[0] + 1), axis=0)
            f['magnetic_z'].resize((f['magnetic_z'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)

            # Save data
            f['pdf'][-1] = distribution
            f['density'][-1] = density
            f['current_v'][-1] = current_v
            f['current_w'][-1] = current_w
            f['electric_x'][-1] = electric_x
            f['electric_y'][-1] = electric_y
            f['electric_z'][-1] = electric_z
            f['magnetic_y'][-1] = magnetic_y
            f['magnetic_z'][-1] = magnetic_z
            f['time'][-1] = time

    def read_data(self):
        # open for reading
        with h5py.File(self.write_filename, 'r') as f:
            time = f['time'][()]
            pdf = f['pdf'][()]
            n = f['density'][()]
            j_v = f['current_v'][()]
            j_w = f['current_w'][()]
            e_x = f['electric_x'][()]
            e_y = f['electric_y'][()]
            e_z = f['electric_z'][()]
            b_y = f['magnetic_y'][()]
            b_z = f['magnetic_z'][()]
        return time, pdf, n, j_v, j_w, e_x, e_y, e_z, b_y, b_z
