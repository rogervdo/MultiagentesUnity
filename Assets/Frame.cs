using System.Collections.Generic;
using UnityEngine;

namespace conexion.Assets.Scripts
{
    // Un frame de la simulación
    public class Dato
    {
        public int time;
        public Dictionary<string, string> lights; // Estados semáforos
        public List<Carro> cars; // Lista de carros
    }

    // Almacena todos los frames
    public class Datos : MonoBehaviour
    {
        public List<Dato> frames = new List<Dato>();
    }
}