using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace conexion.Assets.Scripts
{
    // Recibe datos por TCP
    public class ReceptorTCP : MonoBehaviour
    {
        [Header("Config Red")]
        public int puerto = 1101;

        // TCP
        private TcpListener tcp;
        private Thread hilo;
        private bool activo = false;

        // Datos recibidos
        private List<Dato> datos = null;
        private bool nuevos = false;

        private void Start()
        {
            IniciarTCP();
        }

        private void Update()
        {
            ProcesarDatos();
        }

        private void ProcesarDatos()
        {
            if (!nuevos || datos == null) return;

            // Buscar simulador y pasarle los datos
            var sim = FindFirstObjectByType<Simulator>();
            if (sim != null)
            {
                var mgr = sim.GetComponent<Datos>();
                if (mgr != null)
                {
                    mgr.frames = datos;
                    Debug.Log($"Recibidos {datos.Count} frames");
                }
            }
            nuevos = false;
        }

        private void IniciarTCP()
        {
            tcp = new TcpListener(IPAddress.Any, puerto);
            hilo = new Thread(ManejarRed) { IsBackground = true };
            hilo.Start();
            Debug.Log($"TCP en puerto {puerto}");
        }

        private void ManejarRed()
        {
            try
            {
                tcp.Start();
                activo = true;

                while (activo)
                {
                    using (var cliente = tcp.AcceptTcpClient())
                    {
                        ProcesarJSON(cliente);
                    }
                }
            }
            catch (System.Exception ex)
            {
                if (activo)
                    Debug.LogError("Error TCP: " + ex.Message);
            }
        }

        private void ProcesarJSON(TcpClient cliente)
        {
            NetworkStream stream = cliente.GetStream();

            try
            {
                byte[] header = LeerHeader(stream);
                if (header == null) return;

                int tamano = ExtraerTamano(header);

                byte[] json = LeerCompleto(stream, tamano);
                if (json == null) return;

                string texto = Encoding.UTF8.GetString(json);
                datos = JsonConvert.DeserializeObject<List<Dato>>(texto);

                nuevos = true;
            }
            catch (System.Exception ex)
            {
                Debug.LogError("Error JSON: " + ex.Message);
            }
        }

        private byte[] LeerHeader(NetworkStream stream)
        {
            byte[] buffer = new byte[4];
            int leidos = 0;

            while (leidos < 4)
            {
                int actual = stream.Read(buffer, leidos, 4 - leidos);
                if (actual == 0) return null;
                leidos += actual;
            }

            return buffer;
        }

        private int ExtraerTamano(byte[] header)
        {
            return IPAddress.NetworkToHostOrder(System.BitConverter.ToInt32(header, 0));
        }

        private byte[] LeerCompleto(NetworkStream stream, int tamano)
        {
            byte[] buffer = new byte[tamano];
            int leidos = 0;

            while (leidos < tamano)
            {
                int actual = stream.Read(buffer, leidos, tamano - leidos);
                if (actual == 0) return null;
                leidos += actual;
            }

            return buffer;
        }

        private void OnDestroy()
        {
            Limpiar();
        }

        private void Limpiar()
        {
            activo = false;

            if (hilo?.IsAlive == true)
                hilo.Join(1000);

            tcp?.Stop();
        }
    }
}