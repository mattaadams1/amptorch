import numpy as np

from ase.calculators.calculator import Calculator
from amptorch.utils import make_amp_descriptors_simple_nn
import pandas as pd

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class EnsembleCalc(Calculator):
    """Atomistics Machine-Learning Potential (AMP) ASE calculator
   Parameters
   ----------
    model : object
        Class representing the regression model. Input arguments include training
        images, descriptor type, and force_coefficient. Model structure and training schemes can be
        modified directly within the class.

    label : str
        Location to save the trained model.

    """

    implemented_properties = ["energy", "forces", "uncertainty"]

    def __init__(self, trained_calcs, training_params):
        Calculator.__init__(self)
        self.trained_calcs = trained_calcs
        self.training_params = training_params

    median_list = [100]
    statistics = []
    def calculate_stats(self, energies, forces,energy_list = median_list,stat_list = statistics):
        median_idx = np.argsort(energies)[len(energies) // 2]
        energy_median = energies[median_idx]
        prev_e_mean = energy_list[-1] #previous median energy
        energy_list.append(energy_median)
        #print('previous median',prev_e_mean)
        stat_list.append([np.max(np.var(forces,  axis=0)),np.max(np.std(forces, axis=0))])
        forces_median = forces[median_idx]
        l = []
        def fmax(forces):
            for i in range(len(forces)):
                l.append(np.max(np.absolute(forces[i])))
            return l
           
        fmax = fmax(forces)
        #print('fmax_predict',fmax[median_idx])

        #print(np.max(forces))
        max_forces_var = np.max(np.var(forces, axis=0))
        #print('forces',forces)
        #print(max_forces_var)
        df = pd.DataFrame(stat_list,columns = ['max force var','max force stdev'])
        df.to_csv('energy_force_statistics.csv',index=False)
        return energy_median, forces_median, max_forces_var

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energies = []
        forces = []

        make_fps(atoms, self.training_params["Gs"])
        for calc in self.trained_calcs:
            energies.append(calc.get_potential_energy(atoms))
            forces.append(calc.get_forces(atoms))
        energies = np.array(energies)
        forces = np.array(forces)
        energy_pred, force_pred, uncertainty = self.calculate_stats(energies, forces)

        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
        atoms.info["uncertainty"] = np.array([uncertainty])


def make_fps(images, Gs):
    if isinstance(images, list):
        pass
    else:
        images = [images]
    elements = np.array([atom.symbol for atoms in images for atom in atoms])
    _, idx = np.unique(elements, return_index=True)
    elements = list(elements[np.sort(idx)])

    make_amp_descriptors_simple_nn(
        images, Gs, elements, cores=1, label="ensemble"
    )

