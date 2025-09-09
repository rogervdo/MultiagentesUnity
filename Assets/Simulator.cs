using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace conexion.Assets.Scripts
{
    // Controla la simulación de tráfico
    public class Simulator : MonoBehaviour
    {
        [Header("Configuración")]
        public GameObject prefabCarro;
        public GameObject TLnorte;
        public GameObject TLsur;
        public GameObject TLeste;
        public GameObject TLoeste;
        public float velocidad = 5f; // Frames por segundo
        public float rotacion = 5f;

        // Estado actual
        private float tiempo = 0f;
        private int indice = 0;

        // Diccionarios de carros
        private Dictionary<int, GameObject> carros = new Dictionary<int, GameObject>();
        private Dictionary<int, Vector3> posAnt = new Dictionary<int, Vector3>();
        private Dictionary<int, Vector3> posObj = new Dictionary<int, Vector3>();
        private Dictionary<int, float> progreso = new Dictionary<int, float>();
        private Datos datos;

        private void Awake()
        {
            datos = GetComponent<Datos>();
            if (datos == null)
            {
                Debug.Log("Componente Datos no encontrado, creando uno nuevo...");
                datos = gameObject.AddComponent<Datos>();
            }
        }

        private void Update()
        {
            if (datos?.frames == null || indice >= datos.frames.Count) return;

            // Controlar tiempo entre frames
            tiempo += Time.deltaTime;
            float intervalo = 1f / velocidad;

            if (tiempo >= intervalo)
            {
                tiempo = 0f;
                EjecutarFrame(datos.frames[indice]);
                indice++;
            }

            MoverCarros(); // Suavizar movimiento
        }

        private void MoverCarros()
        {
            float intervalo = 1f / velocidad;
            float paso = Time.deltaTime / intervalo;

            foreach (var id in carros.Keys.ToList())
            {
                if (!progreso.ContainsKey(id)) continue;

                progreso[id] += paso;
                progreso[id] = Mathf.Clamp01(progreso[id]);

                Vector3 inicio = posAnt[id];
                Vector3 fin = posObj[id];
                Vector3 actual = Vector3.Lerp(inicio, fin, progreso[id]);

                carros[id].transform.position = actual;

                Vector3 direccion = (fin - inicio).normalized;
                if (direccion != Vector3.zero)
                {
                    Quaternion rot = Quaternion.LookRotation(direccion);
                    carros[id].transform.rotation =
                        Quaternion.Slerp(carros[id].transform.rotation, rot, Time.deltaTime * rotacion);
                }
            }
        }

        public void EjecutarFrame(Dato frameData)
        {
            if (frameData == null) return;

            // Actualizar semáforos
            if (frameData.lights != null)
                ActualizarSemaforos(frameData.lights);

            // Procesar carros
            if (frameData.cars != null)
            {
                // Track cars that are still active in this frame
                HashSet<int> activeCarIds = new HashSet<int>();

                foreach (var carro in frameData.cars)
                {
                    activeCarIds.Add(carro.id);

                    if (!carros.ContainsKey(carro.id))
                        CrearCarro(carro);
                    PonerObjetivo(carro);

                    // Delete car if it has arrived
                    if (carro.estado == "arrived")
                    {
                        EliminarCarro(carro.id);
                    }
                }

                // Delete cars that are no longer in the frame data
                foreach (var carId in carros.Keys.ToList())
                {
                    if (!activeCarIds.Contains(carId))
                    {
                        EliminarCarro(carId);
                    }
                }
            }
        }

        private void CrearCarro(Carro carro)
        {
            if (prefabCarro == null)
            {
                Debug.LogError("No se puede crear vehículo: ¡prefabCarro es nulo!");
                return;
            }

            GameObject nuevo = Instantiate(prefabCarro);
            Vector3 pos = new Vector3(carro.x, 0, carro.y);
            nuevo.transform.position = pos;

            carros[carro.id] = nuevo;
            posAnt[carro.id] = pos;
            posObj[carro.id] = pos;
            progreso[carro.id] = 1f;
        }

        private void PonerObjetivo(Carro carro)
        {
            Vector3 nuevaPos = new Vector3(carro.x, 0, carro.y);

            if (posObj.ContainsKey(carro.id))
            {
                posAnt[carro.id] = carros[carro.id].transform.position;
            }
            else
            {
                posAnt[carro.id] = nuevaPos;
            }

            posObj[carro.id] = nuevaPos;
            progreso[carro.id] = 0f;
        }

        private void EliminarCarro(int carId)
        {
            if (carros.ContainsKey(carId))
            {
                Destroy(carros[carId]);
                carros.Remove(carId);
                posAnt.Remove(carId);
                posObj.Remove(carId);
                progreso.Remove(carId);
            }
        }

        private void ActualizarSemaforos(Dictionary<string, string> estados)
        {
            CambiarColorSemaforo(TLnorte, estados, "N");
            CambiarColorSemaforo(TLsur, estados, "S");
            CambiarColorSemaforo(TLeste, estados, "E");
            CambiarColorSemaforo(TLoeste, estados, "W");
        }

        private void CambiarColorSemaforo(GameObject semaforo, Dictionary<string, string> estados, string dir)
        {
            if (semaforo == null || !estados.ContainsKey(dir)) return;
            var renderer = semaforo.GetComponent<Renderer>();
            if (renderer == null) return;
            switch (estados[dir])
            {
                case "G": renderer.material.color = Color.green; break;
                case "R": renderer.material.color = Color.red; break;
                default: renderer.material.color = Color.yellow; break;
            }
        }
    }
}